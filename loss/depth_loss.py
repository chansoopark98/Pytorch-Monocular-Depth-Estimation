import torch
import torch.nn as nn
import numpy as np

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()

    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def imgrad_loss(pred, gt, mask=None):
    N,C,_,_ = pred.size()
    grad_y, grad_x = imgrad(pred)
    grad_y_gt, grad_x_gt = imgrad(gt)
    grad_y_diff = torch.abs(grad_y - grad_y_gt)
    grad_x_diff = torch.abs(grad_x - grad_x_gt)
    if mask is not None:
        grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
        grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
    return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))


def scale_invariant_loss(pred, gt):
    logdiff = torch.log(pred) - torch.log(gt)
    scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85*(logdiff.mean() ** 2))*10.0
    return scale_inv_loss

# def imgrad(img):
#     img = torch.mean(img, 1, True)
#     fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
#     conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#     weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
#     if img.is_cuda:
#         weight = weight.cuda()
#     conv1.weight = nn.Parameter(weight)
#     grad_x = conv1(img)

#     fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#     conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#     weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

#     if img.is_cuda:
#         weight = weight.cuda()

#     conv2.weight = nn.Parameter(weight)
#     grad_y = conv2(img)
#     return grad_y, grad_x

# def DSSIM(self, x, y):

#     c1 = 0.01 ** 2
#     c2 = 0.03 ** 2

#     avgpool = nn.AvgPool2d(3, 1)

#     mu_x = [avgpool(x[i]) for i in range(self.n)]
#     mu_y = [avgpool(y[j]) for j in range(self.n)]
#     mu_x_sq = [mu_x[i] ** 2 for i in range(self.n)]
#     mu_y_sq = [mu_y[j] ** 2 for j in range(self.n)]

#     #sigma = E[X^2] - E[X]^2
#     sigma_x = [avgpool(x[i] ** 2) - mu_x_sq[i] for i in range(self.n)]
#     sigma_y = [avgpool(y[j] ** 2) - mu_y_sq[j] for j in range(self.n)]
#     #cov = E[XY] - E[X]E[Y]
#     cov_xy = [avgpool(x[i] * y[i]) - (mu_x[i] * mu_y[i]) for i in range(self.n)]

#     SSIM_top = [(2 * mu_x[i] * mu_y[i] + c1) * (2 * cov_xy[i] + c2) for i in range(self.n)]
#     SSIM_bot = [(mu_x_sq[i] + mu_y_sq[i] + c1) * (sigma_x[i] + sigma_y[i] + c2) for i in range(self.n)]

#     SSIM = [SSIM_top[i] / SSIM_bot[i] for i in range(self.n)]
#     DSSIM = [torch.mean(torch.clamp((1 - SSIM[i]) / 2, 0, 1)) for i in range(self.n)]

#     return DSSIM

# def grad_loss(pred, gt, mask=None):
#     N,C,_,_ = pred.size()
#     grad_y, grad_x = imgrad(pred)
#     grad_y_gt, grad_x_gt = imgrad(gt)
#     grad_y_diff = torch.abs(grad_y - grad_y_gt)
#     grad_x_diff = torch.abs(grad_x - grad_x_gt)
#     if mask is not None:
#         grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
#         grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
#     return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))

# def berhu_loss(valid_out, valid_gt):         
#     diff = valid_out - valid_gt
#     diff_abs = torch.abs(diff)
#     c = 0.2*torch.max(diff_abs.detach())         
#     mask2 = torch.gt(diff_abs.detach(),c)
#     diff_abs[mask2] = (diff_abs[mask2]**2 +(c*c))/(2*c)
#     return diff_abs.mean()