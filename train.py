# This code file is a uniform training framework for CLIPSym and other baseline models.
# The structure borrows from the training of EquiSym: https://github.com/ahyunSeo/EquiSym/blob/master/train.py, which is under MIT License.

import os
import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from tqdm import tqdm
from torchvision import transforms
import torch.utils.data as data
from utils import *
import utils
from config import *
import wandb
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import uuid, time, cv2

from dendi_loader import NewSymmetryDatasets
from models.clipsym import CLIPSym


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(global_seed):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(999)
    np.random.seed(global_seed)
    random.seed(global_seed)

def train(model, args, train_loader, val_loaders, test_loaders, device, start_epoch=1, max_f1=0.0):
    param_groups = model.parameters()
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # mark the start time, for calculating total training time and each epoch's duration 
    time_per_epoch = 0

    # best checkpoint path, for saving and loading
    best_ckpt_path = './weights/v_' + args.ver + '_best_checkpoint.pt'

    if args.rot_data:
        sym_type = 'rotation'
    else:
        sym_type = 'reflection'    
    
    n_thresh = 100

    for epoch in range(start_epoch, args.num_epochs + 1):
        axis_eval = PointEvaluation(n_thresh=n_thresh, blur_pred=True, device=device)
        # if epoch == start_epoch + 1:
        #     time_per_epoch = time.time() - start_time
        adjust_learning_rate(args, optimizer, epoch, lr_type=args.lr_type)
        # wandb log the lr
        if not args.wandb_off:
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
        model.train()
        print('>>> Epoch ', epoch, " training started")
        running_sample = 0
        running_loss = 0
        running_axis_loss = 0
        
        for idx, data in enumerate(tqdm(loaders[train_loader])):
            img, _, axis, axis_gs, _, a_lbl, img_region = data
            img, axis, axis_gs, a_lbl = img.to(device), axis.to(device), axis_gs.to(device), a_lbl.to(device)
            axis_out, total_loss, losses = model(img, axis, axis_gs, a_lbl)
            loss = total_loss.mean()
            running_axis_loss += losses[0].mean().item()

            axis_out = F.interpolate(axis_out, size=axis.size()[2:], mode='bilinear', align_corners=True)
            axis_eval(axis_out, axis)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            running_sample += 1
            running_loss += loss.item()

            if (idx == len(loaders[train_loader]) - 1) and not args.wandb_off:
                wandb.log({'train_total_loss': running_loss / running_sample}, step=epoch)
                
                # log the last batch images to wandb
                if args.track_img_train and epoch % args.wandb_track_img_interval == 0:
                    train_log_images = [
                        wandb.Image(unnorm(img), caption='Image/train_%d_image' % idx),
                        wandb.Image(axis_gs.float(),  caption='Image/train_%d_axisGT' % idx),              
                        wandb.Image(axis_out.cpu(), caption='Image/train_%d_axisPred' % idx),
                    ]
                    wandb.log({'train_log_images': train_log_images}, step=epoch)
                     
                # reset 
                running_sample = 0
                running_loss = 0

        train_rec, train_prec, train_f1 = axis_eval.f1_score()
        print(f"training max f1: {train_f1.max()}")
        
        if not args.wandb_off:
            wandb.log({'train_max_f1': train_f1.max()}, step=epoch)
            wandb.log({'train_max_f1_idx': train_f1.argmax()}, step=epoch)

        if epoch % args.val_interval == 0:
            rec, prec, f1, f1_max = test(model, epoch, args, val_loaders, device, sym_type=sym_type)
            
            _max_f1 = f1_max[0]
            max_f1 = max(max_f1, _max_f1)

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'max_f1': max_f1,
            }
            
            # if model has input_ids and attention_mask, save them to the checkpoint
            if hasattr(model, 'prompts'):
                checkpoint['prompts'] = model.prompts
                checkpoint['input_ids'] = model.input_ids
                checkpoint['attention_mask'] = model.attention_mask
                checkpoint['fixed_cond_embed'] = model.fixed_cond_embed

            torch.save(checkpoint, './weights/v_' + args.ver + '_last_checkpoint.pt')
            print(f'current max f1: {max_f1}')

            if max_f1 == _max_f1:
                print('best model renewed.')
                print(f'best val f1: {max_f1} at epoch {epoch}')
                torch.save(checkpoint, './weights/v_' + args.ver + '_best_checkpoint.pt')
                print('best model saved.')

    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    rec, prec, f1, f1_max  = test(model, ckpt['epoch'], args, test_loaders, device, mode='test', sym_type=sym_type)

    # Add the model running stats to the checkpoint and save it
    checkpoint['stats'] = rec, prec, f1, f1_max
    torch.save(checkpoint, './weights/v_' + args.ver + '_best_checkpoint.pt')

def test(model, epoch, args, test_loaders, device, mode='test', sym_type='reflection'):
    model.eval()
    # n_thresh = 100 if mode in ['test', 'ref_test', 'rot_test'] else 10
    n_thresh = 100

    recs, precs, f1s, f1_maxs = [], [], [], []

    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            print(f">>>> Testing on {test_loader} started.")
            val_sample, val_total_loss, val_axis_loss, val_theta_loss = 0, 0, 0, 0
            axis_eval = PointEvaluation(n_thresh, blur_pred=True, device=device)
            val_log_images = []

            for idx, data in enumerate(tqdm(loaders[test_loader])):
                img, _, axis, axis_gs, _, a_lbl, img_region = data
                
                img, axis, axis_gs, a_lbl = img.to(device), axis.to(device), axis_gs.to(device), a_lbl.to(device)

                axis_out, total_loss, losses = model(img, axis, axis_gs, a_lbl)

                val_axis_loss += losses[0].mean().item()

                if args.get_theta:
                    val_theta_loss += losses[1].mean().item()
                
                if (args.test_resize):
                    img_region = img_region[0]
                    img = img[:, :, img_region[0]:img_region[1], img_region[2]:img_region[3]]
                    axis_out = axis_out[:, :, img_region[0]:img_region[1], img_region[2]:img_region[3]]
                    axis_gs = axis_gs[:, :, img_region[0]:img_region[1], img_region[2]:img_region[3]]
                    
                axis_out = F.interpolate(axis_out, size=axis.size()[2:], mode='bilinear', align_corners=True)
                axis_eval(axis_out, axis)
                img = F.interpolate(img, size=axis.size()[2:], mode='bilinear', align_corners=True)

                val_total_loss += total_loss.mean().item()
                val_sample += 1

                if not args.wandb_off and epoch % args.wandb_track_img_interval == 0:
                    _val_log_images = [
                        wandb.Image(unnorm(img), caption='Image/%s%d_image%d' % (mode, i, idx)),
                        wandb.Image(axis_gs,  caption='Image/%s%d_axisGT%d' % (mode, i, idx)),              
                        wandb.Image(axis_out.cpu(), caption='Image/%s%d_axisPred%d' % (mode, i, idx)),
                    ]
                    val_log_images += _val_log_images
                    if idx == len(loaders[test_loader]) - 1:
                        wandb.log({'%s_log_images' % test_loader: val_log_images}, step=epoch)
                    
            rec, prec, f1 = axis_eval.f1_score()
            recs.append(rec), precs.append(prec), f1s.append(f1), f1_maxs.append(f1.max())
            print(f"{test_loader} max f1: {f1.max()}")

            if not args.wandb_off:
                wandb.log({'%s_total_loss' % test_loader: val_total_loss / val_sample}, step=epoch)
                wandb.log({'%s_max_f1' % test_loader: f1.max()}, step=epoch)
                wandb.log({'%s_max_f1_idx' % test_loader: f1.argmax()}, step=epoch)
        
    return recs, precs, f1s, f1_maxs
                
                

if __name__ == '__main__':
    args = get_parser()
    set_seed(args.seed)

    print("=====================================================================")
    print()
                   
    args.sync_bn = True
    comment = str(args.ver)
    print(">>> version: ", comment)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>> Let's use", torch.cuda.device_count(), "GPUs!")
    print("    Available GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"        Device {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    last_ckpt_path = './weights/v_' + args.ver + '_last_checkpoint.pt'
    checkpoint = None
    if os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path)

    if args.model_name == 'pmcnet':
        from models.pmcnet import PMCNet
        import models.pmc_config as pmc_config
        pmc_args = pmc_config.get_parser()
        pmc_args = pmc_config.check_ablation(pmc_args)
        model = PMCNet(pmc_args)
    if args.model_name == 'equisym':
        from models.equisym import EquiSym
        model = EquiSym(args)
    if args.model_name == 'clipsym':
        prompts = utils.get_prompts(args)
        model = CLIPSym(args, prompts, reduce_dim=args.reduce_dim)
    # if args.model_name == 'notext':
    #     from models.clipsym_notext import CLIPSym_notext
    #     model = CLIPSym_notext(args, reduce_dim=args.reduce_dim)
        
    print(f'>>> args.model_name: {args.model_name} is defined.\n')
    model = model.to(device)

    if not args.wandb_off:
        print(">>> wandb on")
        wandb.init(id=comment, project=args.project_name, notes=comment, save_code=True, resume=True)
        wandb.config.update(args, allow_val_change=True)
    else:
        print(">>> wandb off")

    if args.rot_data:
        sym_type = 'rotation'
    else:
        sym_type = 'reflection'
    
    if args.dataset == 'dendi':
        dendi_trainset = NewSymmetryDatasets(sym_type=sym_type, split='train', input_size=[args.input_size, args.input_size], get_theta=args.get_theta, with_ref_circle=True, t_resize=False)
        dendi_valset = NewSymmetryDatasets(sym_type=sym_type, split='val', input_size=[args.input_size, args.input_size], get_theta=args.get_theta, with_ref_circle=True, t_resize=args.test_resize)
        dendi_testset = NewSymmetryDatasets(sym_type=sym_type, split='test', input_size=[args.input_size, args.input_size], get_theta=args.get_theta, with_ref_circle=True, t_resize=args.test_resize)
        dendi_train_loader = data.DataLoader(dendi_trainset, batch_size=args.bs_train, shuffle=True, drop_last=True)
        dendi_val_loader = data.DataLoader(dendi_valset, batch_size=1, shuffle=False)
        dendi_test_loader = data.DataLoader(dendi_testset, batch_size=1, shuffle=False)
        
        loaders = {'train': dendi_train_loader, 'val': dendi_val_loader, 'test': dendi_test_loader}
    elif args.dataset == 'pmc':
        from pmc_loader import SymmetryDatasets
        DATA_DICT = {'n':'SYM_NYU', 'l': 'SYM_LDRS', 's': 'SYNTHETIC_COCO', 'd': 'SYM_SDRW'}
        train_data = list(args.pmc_train_data)
        train_data = [DATA_DICT[data] for data in train_data]
        
        # PMC sets and loaders
        pmc_trainset = SymmetryDatasets(dataset=train_data, split='train', input_size=[args.input_size, args.input_size])
        pmc_train_loader = data.DataLoader(pmc_trainset, batch_size=args.bs_train, shuffle=True, num_workers=4, drop_last=True)
        pmc_valset = SymmetryDatasets(dataset=['SYM_SDRW', 'SYM_LDRS'], split='val', input_size=[args.input_size, args.input_size], samescale=args.test_resize)
        pmc_val_loader = data.DataLoader(pmc_valset, batch_size=1, shuffle=False, num_workers=4)

        
        test_sdrw = SymmetryDatasets(dataset=['SYM_SDRW'], split='test', input_size=[args.input_size, args.input_size], samescale=args.test_resize)
        sdrw_test_loader = data.DataLoader(test_sdrw, batch_size=1, shuffle=False, num_workers=4)
        test_ldrs = SymmetryDatasets(dataset=['SYM_LDRS'], split='test', input_size=[args.input_size, args.input_size], samescale=args.test_resize)
        ldrs_test_loader = data.DataLoader(test_ldrs, batch_size=1, shuffle=False, num_workers=4)
        test_mixed = SymmetryDatasets(dataset=['SYM_LDRS', 'SYM_SDRW'], split='test', input_size=[args.input_size, args.input_size], samescale=args.test_resize)
        mixed_test_loader = data.DataLoader(test_mixed, batch_size=1, shuffle=False, num_workers=4)
        loaders = {'train': pmc_train_loader, 'val': pmc_val_loader, 'test_sdrw': sdrw_test_loader, 'test_ldrs': ldrs_test_loader, 'test_mixed': mixed_test_loader}

    start_epoch = 1
    max_f1 = 0.0
    if args.continue_training and checkpoint is not None:
        print('continue training')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        start_epoch = checkpoint['epoch'] + 1
        max_f1 = checkpoint['max_f1']
        print('start_epoch', start_epoch)
        print('max_f1', max_f1)
    
    if args.dataset == 'dendi':
        train(model, args, 'train', ('val', 'test'), ('test', ), device, start_epoch=start_epoch, max_f1=max_f1)
    elif args.dataset == 'pmc':
        train(model, args, 'train', ('val', 'test_sdrw', 'test_ldrs', 'test_mixed'), ('test_sdrw', 'test_ldrs', 'test_mixed'), device, start_epoch=start_epoch, max_f1=max_f1)
        

            


        




