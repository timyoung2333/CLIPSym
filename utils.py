# Used parts of https://github.com/ahyunSeo/EquiSym/blob/master/utils.py for data processing and model evaluation. The code is under MIT license.

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

def get_prompts(args):
    import pandas as pd
    objects = pd.read_csv('./sym_datasets/objects.csv', header=None)
    if args.prompts_type == 'group':
        prompts = []
        for _ in range(args.num_prompts):
            prompt = objects.sample(args.words_per_prompt)[0].tolist()
            prompts.append(' '.join(prompt))
    elif args.prompts_type == 'fully_random':
        # generate args.num_prompts random prompts from alphabets
        prompts = [' '.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=args.words_per_prompt)) for _ in range(args.num_prompts)]
    elif args.prompts_type == 'ref_axis':
        prompts = ['reflection axis']
    elif args.prompts_type == 'sym_axis':
        prompts = ['symmetry axes in the image']
    
    return prompts

    
class DFS:
    def __init__(self, E, S):
        self.E = E
        self.S = S
        self.visited = np.zeros(len(S)).astype(bool)
        self.state = []
        self.score = 0
        self.max_state = []
        self.max_score = 0
        self.max_visited = 0
        
    def dfs(self):
        if (self.max_score < self.score * self.visited.sum()):
            del self.max_score
            self.max_state = self.state.copy()
            self.max_visited = self.visited.sum()
            self.max_score = self.score * self.max_visited
        for j in range(len(self.S)):
            if self.visited[j]: continue
            flag = 1
            for k in self.state:
                if not self.E[j][k]: flag = 0
            if flag:
                self.visited[j]=True
                self.state.append(j)
                self.score += self.S[j]
                self.dfs()
                self.state.pop()
                self.score -= self.S[j]
                self.visited[j]=False
                
    def forward(self):
        self.dfs()
        return self.max_state
    
def draw_axis(lines, size):
    axis = Image.new('L', size)
    # w, h = img.size
    draw = ImageDraw.Draw(axis)
    length = np.array([size[0], size[1], size[0], size[1]])
    
    # x1, y1, x2, y2
    line_coords = []

    for idx, coords in enumerate(lines):
        if coords[0] > coords[2]:
            coords = np.roll(coords, -2)
        draw.line(list(coords), fill=(idx + 1))
        coords = np.array(coords).astype(np.float32)
        _line_coords = coords / length
        line_coords.append(_line_coords)
    axis = np.asarray(axis).astype(np.float32)
    return axis, line_coords

def match_input_type(img):
    img = np.asarray(img)
    if img.shape[-1] != 3:
        img = np.stack((img, img, img), axis=-1)
    return img 
    
def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = (img - mean) / std
    return img

def unnorm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = img * std.to(img.device) + mean.to(img.device)
    return img

##########################
### Train ################
##########################

def sigmoid_focal_loss(
    source: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
    is_logits=True
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if is_logits:
        p = nn.Sigmoid()(source)
        ce_loss = F.binary_cross_entropy_with_logits(
            source, targets, reduction="none"
        )
    else:
        p = source
        ce_loss = F.binary_cross_entropy(source, targets, reduction="none")
    
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
    
def adjust_learning_rate(args, optimizer, epoch, lr_type='step'):
    if lr_type == 'constant':
        lr = args.lr
        
    if lr_type == 'step':
        lr = args.lr * (0.1 ** (epoch // (args.num_epochs * 0.5))) * (0.1 ** (epoch // (args.num_epochs * 0.75)))
    
    if lr_type == 'exp':
        lr = args.lr * (args.lr_decay ** (epoch / args.num_epochs))
    
    if lr_type == 'cos':
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            min_lr = args.lr * args.lr_decay
            decay = 0.5 * (1 + np.cos((epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs) * np.pi))
            lr = min_lr + (args.lr - min_lr) * decay
    
    if lr_type == 'cos_cyclic':
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            min_lr = args.lr * args.lr_decay
            T = (args.num_epochs - args.warmup_epochs) / args.cyclic_n # 100
            passed_cycles = (epoch - args.warmup_epochs - 1) // T # epoch = 500, passed_cycles = 4
            decay = 0.5 * (1 + np.cos((epoch - args.warmup_epochs - passed_cycles * T) / T  * np.pi))
            lr = min_lr + (args.lr - min_lr) * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##########################
### Evaluation ###########
##########################

def make_gt_filter(max_dist):
    # expand 1-pixel gt to a circle w/ radius of max_dist
    ks = max_dist * 2 + 1
    filters = torch.zeros(1, 1, ks, ks)
    for i in range(ks):
        for j in range(ks):
            dist = (i-max_dist) ** 2 + (j-max_dist) ** 2
            if dist <= max_dist**2:
                filters[0, 0, i, j] = 1
    return filters

class PointEvaluation(object):
    def __init__(self, n_thresh=100, max_dist=5, blur_pred=False, device=None):
        self.n_thresh = n_thresh
        self.max_dist = max_dist
        self.thresholds = torch.linspace(1.0 / (n_thresh + 1),
                            1.0 - 1.0 / (n_thresh + 1), n_thresh)
        if device is not None:
            self.filters = make_gt_filter(max_dist).to(device)
        else:
            self.filters = make_gt_filter(max_dist)
        self.tp = torch.zeros((n_thresh,))
        self.pos_label = torch.zeros((n_thresh,))
        self.pos_pred = torch.zeros((n_thresh,))
        self.num_samples = 0
        self.blur_pred = blur_pred
        self.mse = 0
        self.iou = 0
    
    def f1_score(self,):
        # If there are no positive pixel detections on an image,
        # the precision is set to 0.
        precision = torch.where(self.pos_pred > 0, self.tp / self.pos_pred, torch.zeros(1))
        recall = self.tp / self.pos_label
        numer = precision + recall
        f1 = torch.where(numer > 0, 2 * precision * recall / numer, torch.zeros(1))
        return precision, recall, f1
        
    def __call__ (self, pred, gt, consistency=False, mask=None, eps=1e-4, filter=True):
        pred = pred.detach()
        gt = gt.to(pred.device)
        
        # # plot pred and gt
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax1.imshow(pred[0, 0].cpu().numpy())
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(gt[0, 0].cpu().numpy())
        # plt.show()
        # plt.savefig('pred_gt.png')
        
        # img, img_rot
        # img_rot = rotate(img, angle)
        # pred = model(img_rot)
        # gt = rotate(model(img), angle)
        # F1-consistency = F1(pred, gt>th, th)
        # pred_{x, y} = 0.2, gt_{x, y} = 0.8, th = 0.15,
        
        if filter:
            gt_new = F.conv2d(gt, self.filters, padding=self.max_dist)
        gt_new = (gt_new > eps).float()
        pos_label = gt_new.float().sum(dim=(2, 3)).cpu()
        self.num_samples = self.num_samples + pred.shape[0]
        iou = torch.zeros((self.n_thresh,))
        # Evaluate predictions (B, 1, H, W)
        for idx, th in enumerate(self.thresholds):
            _pred = (pred > th).float() # binary

            if self.blur_pred:
                _pred = F.conv2d(_pred, self.filters, padding=self.max_dist)
                _pred = (_pred > eps).float()

            if consistency:
                gt_new = (gt > th).float()
                if self.blur_pred:
                    gt_new = F.conv2d(gt_new, self.filters, padding=self.max_dist)
                    gt_new = (gt_new > eps).float()
                pos_label = gt_new.float().sum(dim=(2, 3)).cpu()
                
            if mask is not None:
                gt_new = gt_new * mask
                _pred = _pred * mask
                
            if (gt_new.max() != 0 or  _pred.max() != 0):
                iou[idx] = (gt_new * _pred).sum().cpu() / ((gt_new + _pred) > 0).float().sum().cpu()
            tp = ((gt_new * _pred) > eps).float().sum(dim=(2, 3))
            pos_pred = _pred.sum(dim=(2, 3)).cpu()
            
            self.tp[idx] += tp.sum().cpu()
            self.pos_pred[idx] += pos_pred.sum()
            self.pos_label[idx] += pos_label.sum()
            
        if consistency:
            self.mse = F.mse_loss(pred, gt)
            
            
        self.iou = iou.max()
            
            