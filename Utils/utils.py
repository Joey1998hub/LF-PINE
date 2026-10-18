import numpy as np;import torch
import torch.nn.functional as F
from torch.fft import fft2,ifft2
import tifffile as tf
import os

def calBacknoise(img,bins=256):
    hist_max = torch.mean(img).item()
    hist_min = torch.min(img).item()
    hist = torch.histc(img,bins,hist_min,hist_max)
    backnoise_index = torch.max(hist,0)[1]
    backnoise = (backnoise_index/bins)*(hist_max-hist_min)+hist_min

    return backnoise

def GenMixedNoisyLF(input, target):
    # size: [B,V,H,W]
    if input.ndim == 3: input = torch.unsqueeze(input,dim=0)
    if target.ndim == 3: target = torch.unsqueeze(target,dim=0)
    Nnum = input.shape[-3]

    input = F.softplus(input,threshold=1)
    signal = input.clone().detach()

    for i in range(Nnum):
        clean = signal[:,i,...]; noisy = target[:,i,...]
        noise = noisy-clean # [B,1,H,W]

        miu = calBacknoise(noisy)
        dark = noisy.clone(); dark[dark>miu] = miu
        count = (noisy<=miu).sum().item()
        var_gs = torch.sum((dark-miu)**2)/count

        var = F.softplus(torch.mean((noise)**2)-var_gs,threshold=1)
        scale = torch.mean(signal[:,i,...])/(var+1e-6)

        input[:,i,...] = input[:,i,...]*scale
        input[:,i,...] = torch.poisson(input[:,i,...])
        input[:,i,...] = input[:,i,...] / scale

    return input

def generate_fps(psf,sample,fp_res=None):
    """
    sample size: z,x,y
    psf size: u,z,x,y
    """
    if sample.ndim == 3: sample = sample.unsqueeze(0).unsqueeze(0)
    if sample.ndim == 4: sample = sample.unsqueeze(1)
    t,_,z,ra,ca = sample.shape
    u,_,rb,cb =psf.shape

    r = ra+rb-1
    p1 = (r-ra)/2

    a1 = torch.zeros([t,1,z,r,r],device=sample.device)
    b1 = torch.zeros([1,u,z,r,r],device=sample.device)

    a1[...,0:ra,0:ca] = sample
    b1[...,0:rb,0:cb] = torch.unsqueeze(psf,0)
    conv1 = ifft2(fft2(a1)*fft2(b1))
    projections = torch.abs(conv1[...,int(p1):int(r-p1),int(p1):int(r-p1)]).squeeze(0)
    if fp_res is not None and projections.shape[-1]!=fp_res: projections = upsample(projections,size=fp_res)
    return projections

def generate_fps_fft(sample_buffer,psf_fft_buffer,sample_res,fp_res=None):
    """
    sample_buffer size: t,1,z,r,r, padded sample buffer
    psf_fft_buffer size: 1,u,z,r,r, FFT of padded PSF buffer
    sample_res: output crop size before optional fp_res resizing
    """
    r = sample_buffer.shape[-1]
    p1 = (r-sample_res)/2

    conv1 = ifft2(fft2(sample_buffer)*psf_fft_buffer)
    projections = torch.abs(conv1[...,int(p1):int(p1+sample_res),int(p1):int(p1+sample_res)]).squeeze(0)
    if fp_res is not None and projections.shape[-1]!=fp_res: projections = upsample(projections,size=fp_res)
    return projections

def generate_psfs_fft(psfs,sample_res):
    """
    psfs size: u,z,x,y
    sample_res: sample x/y size before padding
    """
    u,z,psf_res,_ = psfs.shape
    fft_res = sample_res+psf_res-1
    psfs_fft = torch.zeros((u,z,fft_res,fft_res),device=psfs.device,dtype=torch.complex64)
    padded_psf = torch.zeros((z,fft_res,fft_res),device=psfs.device,dtype=psfs.dtype)
    with torch.no_grad():
        for angle_id in range(u):
            padded_psf.zero_()
            padded_psf[:,0:psf_res,0:psf_res] = psfs[angle_id]
            psfs_fft[angle_id] = fft2(padded_psf)
    return psfs_fft, fft_res

def load_psfs(PSF_dir,Nnum):

    psfs = []
    postfix = os.listdir(PSF_dir)[0].split('.')[1]
    for index in range(Nnum):
        psf_path = os.path.join(PSF_dir,'psf_'+str(index+1)+'.'+postfix)
        psf = torch.from_numpy(tf.imread(psf_path).astype(np.float32))
        psfs.append(psf)
        print('Load PSF:',psf_path,'PSF max:',torch.max(psf).item(),end='\r')
    print('\nLoad %d PSFs.'%len(psfs))
    psfs = torch.stack(psfs,dim=0).squeeze()

    return psfs

def Adjust_lr(optimizer,iter,decay_every,lr_decay,lr_init):

    lr = lr_init * (lr_decay ** (iter // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Learning Rate:',lr)

    return optimizer

def save_results(xguess,save_dir,save_name):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir,save_name)
    tf.imwrite(save_path,xguess.detach().cpu().numpy().astype(np.float32))

def upsample(img,dims=2,factor=None,size=None,mode=None):
    if dims==2 and mode==None: mode = 'bicubic'
    elif dims==3 and mode==None: mode = 'trilinear'
    if len(img.shape) == dims:
        img = torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0)
        if size == None: img = F.interpolate(img,scale_factor=factor, mode=mode)
        else: img = F.interpolate(img,size=size, mode=mode)
        img = img[0,0,:,:]
    elif len(img.shape) == (dims+1):
        img = torch.unsqueeze(img,dim=0)
        if size == None: img = F.interpolate(img,scale_factor=factor, mode=mode)
        else: img = F.interpolate(img,size=size, mode=mode)
        img = img[0,:,:,:]
    elif len(img.shape) == (dims+2):
        if size == None: img = F.interpolate(img,scale_factor=factor, mode=mode)
        else: img = F.interpolate(img,size=size, mode=mode)
    else: raise ValueError('Invalid dims')

    return img
