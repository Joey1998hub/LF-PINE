from torch.fft import fft2
from torch.nn.functional import l1_loss,relu
import torch

def zloss(xguess):

    zloss = torch.mean(torch.abs(2*xguess[1:-1,...]-xguess[:-2,:,:]-xguess[2:,:,:]))
    return zloss

def posloss(xguess):

    return torch.sum(relu(-xguess))


def nllp_loss(fp, lf):

    eps = 1e-8
    loss = torch.mean(fp - lf * torch.log(fp + eps))

    return loss

def fftloss(x,y):
    conv_x = fft2(x)
    conv_y = fft2(y)
    loss = l1_loss(conv_x,conv_y)
    return loss
