import os
import time
import argparse

import torch.nn as nn
import torch.backends
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from utils.logger import TermLogger, AverageMeter
from utils.utils import save_path_formatter, save_checkpoint, tensor2array, json_out, set_all_random_seed
from utils.metrics import compute_errors
from utils import custom_transform
from loss.depth_loss import berhu_loss
from loss.dense_depth_loss import SSIM
from loss.dense_depth_loss import depth_loss as gradient_criterion

from dataset.NyuDataset import DataSequence as dataset
import models

import numpy as np

parser = argparse.ArgumentParser(description='Depth Training scripts',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefix', type=str, 
                    default='test', help='dataset root path')
parser.add_argument('--dataset_root', type=str, 
                    default='./data/', help='dataset root path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0.000001, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--view-freq', default=100, type=int, dest='view_freq',
                    metavar='N', help='view frequency')
parser.add_argument('-e', '--debug', dest='debug', action='store_true',
                    help='debug =======')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

n_iter = 0
val_iter = 0
best_error = -1
start_epoch = 0
args = parser.parse_args()  

def main():
    global best_error, n_iter, device, start_epoch
    # Set Random Seed
    set_all_random_seed(args.seed)
    
    disp_alpha, disp_beta = 10, 0.01
    args.data = os.path.join(args.dataset_root, 'nyu_depth')
    print(f'{args.data}')

    args.img_width = 640
    args.img_height = 480
    
    save_path = save_path_formatter(args, parser)
    args.save_path = f'./saved_models/{save_path}'
    os.makedirs(args.save_path, exist_ok=True)

    print(f'=> will save everything to {args.save_path}')
    
    train_writer = SummaryWriter(args.save_path)   

    json_out(vars(args), args.save_path, 'config.json')

    # Set Data Transforms (Data Augmentation)
    train_transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.AugmentImagePair(),
        custom_transform.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
    ])

    val_transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])])

    # Set train & validation Datasets
    train_set = dataset(root=args.data, 
                        seed=0, 
                        train=True, 
                        shuffle=True,
                        transform=train_transform,
                        scene='train',
                        image_width=args.img_width,
                        image_height=args.img_height)
    
    val_set = dataset(root=args.data, 
                      seed=0, 
                      train=True, 
                      shuffle=False,
                      transform=val_transform,
                      scene='validation',
                      image_width=args.img_width,
                      image_height=args.img_height)
   
    print(f'Train Dataset Samples : {len(train_set)}')
    print(f'Validation Dataset Samples : {len(val_set)}')
    
    # Set DataLoaders from dataset
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # Load Depth Model
    depth_net = models.DepthDecoder(alpha=10., beta=0.1).to(device)
    # Set Model Data Parallel
    depth_net = torch.nn.DataParallel(depth_net)

    # Set optimizer parameters
    optim_params = [
        {'params': depth_net.parameters(), 'lr': args.lr},
    ]
    
    # Set optimizer
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer,
                                                      total_iters=args.epoch_size,
                                                      power=0.9)
    # Terminal Logger
    logger = TermLogger(n_epochs=args.epochs,
                        train_size=min(len(train_loader),args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)     
        logger.reset_train_bar()
        # train
        train_loss = train(args,
                           train_loader,
                           depth_net,
                           optimizer,
                           train_writer,
                           logger,
                           epoch)
        
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # validation
        logger.reset_valid_bar()

        val_loss, _ = validate(args,
                               val_loader,
                               depth_net,
                               train_writer,
                               logger,
                               epoch)
        temp_error = val_loss
        train_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': temp_error}, epoch)

        # LR Update
        scheduler.step()

        if best_error < 0:
            best_error = temp_error

        is_best = temp_error <= best_error
        best_error = min(best_error, temp_error)

        logger.valid_writer.write('* Chkpt: temp_error {:.4f}, mini_error {:.4f}'.format(temp_error, best_error))
    
        save_checkpoint(args.save_path,
                        {'epoch': epoch+1,
                         'state_dict': depth_net.module.state_dict()},
                          is_best)
    logger.epoch_bar.finish()

def train(args, train_loader, disp_net, optimizer, train_writer: SummaryWriter, logger, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    end = time.time()
    global n_iter, device

    ssim = SSIM()
    ssim.to(device)

    disp_net.train()

    for i, (image, depth) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        image = image.to(device)
        depth = depth.to(device)

        preds = disp_net(image)

        ssim_loss = 0
        l1_loss = 0
        gradient_loss = 0


        for pred in preds:
            b, c, h, w = pred.shape
            resized_depth = nn.functional.interpolate(
                depth, (h, w), mode='nearest')

            l1_loss = torch.abs(resized_depth - pred)
            l1_loss = l1_loss.mean()

            ssim_loss = ssim(pred, resized_depth).mean()
            
            gradient_loss += gradient_criterion(resized_depth, pred, device=device).mean()

        net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * gradient_loss)
                + (0.1 * l1_loss)
            )
        # loss = berhu + (l1 * 0.1)

        losses.update(net_loss.data.item(), args.batch_size)
        net_loss.backward()
        optimizer.step()

        n_iter += 1
        batch_time.update(time.time() - end)
        end = time.time()
        logger.train_bar.update(i+1)
        if i % 3 == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {} Epoch {}'.format(batch_time, data_time, losses, epoch))
        if i >= args.epoch_size - 1:
            break

        # Train Writer
        if args.view_freq > 0 and n_iter % args.view_freq == 0:
            train_writer.add_scalar('Train total loss', net_loss.data.item(), n_iter)
            train_writer.add_scalar('Train ssim_loss', ssim_loss.item(), n_iter)
            train_writer.add_scalar('Train gradient_loss', gradient_loss.item(), n_iter)
            train_writer.add_scalar('Train l1_loss', l1_loss.item(), n_iter)
            
            
            inverse_depth = preds[0][0]
            # Inverse Depth to Normal Depth
            normal_depth = 1/ inverse_depth

            train_writer.add_image(tag='Train Depth Inverse Output',
                                img_tensor=tensor2array(inverse_depth, max_value=None, colormap='magma'),
                                global_step=n_iter)

            train_writer.add_image(tag='Train Depth Normal Output',
                                img_tensor=tensor2array(normal_depth, max_value=None, colormap='magma'),
                                global_step=n_iter)
        
    return losses.avg[0]

def voloss(pose1, pose2, k=10):
    if pose1.shape[1] == 6:
        rot1, tra1 = pose1[:, :3], pose1[:, 3:6]
        rot2, tra2 = pose2[:, :3], pose2[:, 3:6]
    elif pose1.shape[2] == 6:
        rot1, tra1 = pose1[:, :, :3], pose1[:, :, 3:6]
        rot2, tra2 = pose2[:, :, :3], pose2[:, :, 3:6]
    return k*(rot1 - rot2).abs().mean() + (tra1 - tra2).abs().mean()

@torch.no_grad()
def validate(args, val_loader, disp_net, train_writer, logger, epoch=0):
    global device
    global val_iter

    batch_time = AverageMeter()
    losses = AverageMeter(precision=5)
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    rmse_log_list = []
    a1_list = []
    a2_list = []
    a3_list = []
    # Calc depth errors
    
    
    for i, (image, depth) in enumerate(val_loader):
        image = image.to(device)
        depth = depth.to(device)

        pred = disp_net(image)

        berhu = 0
        l1 = 0
    
        _, _, h, w = pred.shape
        resized_depth = nn.functional.interpolate(
            depth, (h, w), mode='nearest')
        
        l1 += nn.functional.l1_loss(pred, resized_depth)
        berhu += berhu_loss(pred, resized_depth)

        loss = berhu + (l1 * 0.1)

        losses.update(loss, args.batch_size)
     
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)

        # Compute Original Depth
        normal_gt = 1. / resized_depth
        normal_pred = 1. / pred

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt=normal_gt, pred=normal_pred)
        
        abs_rel_list.append(abs_rel)
        sq_rel_list.append(sq_rel)
        rmse_list.append(rmse)
        rmse_log_list.append(rmse_log)
        a1_list.append(a1)
        a2_list.append(a2)
        a3_list.append(a3)

        if i % 3 == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
            inverse_depth = pred[0]
            normal_depth = 1. / pred[0]

            train_writer.add_image(tag='Valid Depth Inverse Output',
                                img_tensor=tensor2array(inverse_depth, max_value=None, colormap='magma'),
                                global_step=n_iter)
            
            train_writer.add_image(tag='Valid Depth Normal Output',
                                img_tensor=tensor2array(normal_depth, max_value=None, colormap='magma'),
                                global_step=n_iter)
        

    abs_rel_list = np.array(abs_rel_list).mean()
    sq_rel_list = np.array(sq_rel_list).mean()
    rmse_list = np.array(rmse_list).mean()
    rmse_log_list = np.array(rmse_log_list).mean()
    a1_list = np.array(a1_list).mean()
    a2_list = np.array(a2_list).mean()
    a3_list = np.array(a3_list).mean()

    train_writer.add_scalar('Valid abs rel', abs_rel_list, n_iter)
    train_writer.add_scalar('Valid square rel', sq_rel_list, n_iter)
    train_writer.add_scalar('Valid rmse', rmse_list, n_iter)
    train_writer.add_scalar('Valid rmse_log', rmse_log_list, n_iter)
    train_writer.add_scalar('Valid a1', a1_list, n_iter)
    train_writer.add_scalar('Valid a2', a2_list, n_iter)
    train_writer.add_scalar('Valid a3', a3_list, n_iter)

    train_writer.add_scalar('Valid Total loss', loss.item(), n_iter)
    train_writer.add_scalar('Valid berhu loss', berhu.item(), n_iter)
    train_writer.add_scalar('Valid l1 loss', l1.item(), n_iter)
            
            

    logger.valid_bar.update(len(val_loader))

    return losses.avg[0], 0
 

if __name__ == "__main__":
    main()