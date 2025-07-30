import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='SymDet arguments')
    parser.add_argument('--project_name', default='SymDet', type=str)
    parser.add_argument('--input_size', default=417, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--bs_train', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--bs_val', '-bs', default=1, type=int,
                        help='Batch size for validation')
    parser.add_argument('--ver', default='init', type=str)
    parser.add_argument('--lr',         type=float, default=1e-5,      metavar='LR',
                        help="base learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0,        metavar='DECAY',
                        help="weight decay, if 0 nothing happens")
    parser.add_argument('-t', '--test_only', action='store_true')
    parser.add_argument('-wf', '--wandb_off', action='store_true', default=False)
    parser.add_argument('--sync_bn', action='store_true', default=False)

    # e2cnn backbone variants
    parser.add_argument('-eq', '--eq_cnn', action='store_true', default=False) # for equisym
    parser.add_argument('-bb', '--backbone', default='resnet', type=str)
    parser.add_argument('-res', '--depth', default=50, type=int)
    
    parser.add_argument('--dataset', default='dendi', type=str) # dendi, pmc
    parser.add_argument('--pmc_train_data', default='nld', type=str)

    parser.add_argument('-gt', '--get_theta', default=0, type=int)
    parser.add_argument('--n_angle', default=8, type=int)
    parser.add_argument('-rot', '--rot_data', default=0, type=int)
    parser.add_argument('-load_eq', '--load_eq_pretrained', default=1, type=int)
    parser.add_argument('--eq_model_dir', default='./weights/re_resnet50_custom_d8_batch_512.pth', type=str)
    parser.add_argument('--n_rot', default=21, type=int)
    parser.add_argument('-tlw', '--theta_loss_weight', default=1e-3, type=float) # 1e-2 ref, 1e-3 rot

    # Below are my added args
    parser.add_argument('-theta_ls', '--theta_loss_scale', default=1, type=float)
    parser.add_argument('-cont', '--continue_training', default=1, type=int)
    parser.add_argument('--num_prompts', type=int, default=25)
    parser.add_argument('--words_per_prompt', type=int, default=4)
    parser.add_argument('--prompts_type', type=str, default='group')
    
    parser.add_argument('--track_img_train', action='store_true', default=False)
    parser.add_argument('--wandb_track_img_interval', type=int, default=5)
    parser.add_argument('--track_img_test', action='store_true', default=False)
    parser.add_argument('--fl_alpha', type=float, default=0.95)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_type', type=str, default='exp')
    parser.add_argument('--warmup_epochs', type=int, default=20) # for cosine lr and cos_cyclic lr
    
    parser.add_argument('--model_name', type=str, default='clipsym')
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False)
    parser.add_argument('--freeze_img_encoder', action='store_true', default=False)
    parser.add_argument('--clip_scratch', action='store_true', default=False)
    
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--test_resize', action='store_true', default=False)
    parser.add_argument('--reduce_dim', type=int, default=64)

    parser.add_argument('--decoder_linear_map', action='store_true', default=False)
    parser.add_argument('-eq_up', '--equivariant_upsampler', action='store_true', default=False)
    parser.add_argument('--group_fineness', type=int, default=8)

    parser.add_argument('--eval_model', default='pmc', type=str)
    parser.add_argument('--eval_type', default='perf', type=str) # f1, consistency, visualize
    parser.add_argument('--eval_save_gt_only', action='store_true', default=False)
    parser.add_argument('--pre_data', type=str, default='imagenet') # imagenet or clip
    parser.add_argument('--tf_type', type=str, default='rot')
    args = parser.parse_args()

    return args
