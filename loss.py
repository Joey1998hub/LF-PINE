from torch.fft import fft2,ifft2
from torch.nn.functional import l1_loss,relu,mse_loss
import torch.nn.functional as F
import Utils.utils as ut
import torch


def zloss(xguess):

    zloss = torch.mean(torch.abs(2*xguess[1:-1,...]-xguess[:-2,:,:]-xguess[2:,:,:]))
    return zloss

def sparse_tv(xguess):
    loss = torch.mean((xguess)**2)
    return loss

def xy_tv(x):
    if x.ndim>3: x = torch.squeeze(x)
    h_tv = torch.pow((x[:,1:,:]-x[:,:-1,:]),2).mean()
    w_tv = torch.pow((x[:,:,1:]-x[:,:,:-1]),2).mean()
    return h_tv+w_tv

def posloss(xguess):

    return torch.sum(relu(-xguess))

def fftloss(x,y):
    conv_x = fft2(x)
    conv_y = fft2(y)
    loss = l1_loss(conv_x,conv_y)
    return loss