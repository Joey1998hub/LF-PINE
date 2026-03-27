import numpy as np;import torch
import torch.nn.functional as F
from torch.fft import fft2,ifft2
import tifffile as tf
import os;import math

def calBacknoise(img,bins=100):
    hist_max = torch.mean(img).item()
    hist_min = torch.min(img).item()
    hist = torch.histc(img,bins,hist_min,hist_max)
    backnoise_index = torch.max(hist,0)[1]+1
    backnoise = (backnoise_index/bins)*(hist_max-hist_min)+hist_min

    return backnoise

def GenNoisyLF(input, target):
    # size: [B,V,H,W]
    if input.ndim == 3: input = torch.unsqueeze(input,dim=0)
    if target.ndim == 3: target = torch.unsqueeze(target,dim=0)
    Nnum = input.shape[-3]

    signal = input.clone().detach()
    signal = F.softplus(signal,threshold=1e-2)
    poisson_noise = torch.zeros_like(signal)

    for i in range(Nnum):
        clean = signal[:,i,...]; noisy = target[:,i,...]
        var = torch.mean((noisy-clean)**2)
        # scale = torch.mean(signal[:,i,...])/(var+1e-6)
        scale = torch.mean(noisy)/(var+1e-6)

        poisson_noise[:,i,...] = torch.poisson(signal[:,i,...]*scale)/scale - signal[:,i,...]

    return input+poisson_noise

def GenMixedNoisyLF(input, target):
    # size: [B,V,H,W]
    if input.ndim == 3: input = torch.unsqueeze(input,dim=0)
    if target.ndim == 3: target = torch.unsqueeze(target,dim=0)
    Nnum = input.shape[-3]

    input = F.softplus(input,threshold=1)
    signal = input.clone().detach()

    for i in range(Nnum):
        clean = signal[:,i,...]; noisy =  target[:,i,...]
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

#MicroNeRF utils
def generate_fp(psf,sample,fp_res=None):
    z,ra,ca = sample.shape
    _,rb,cb =psf.shape

    r = ra+rb-1
    p1 = (r-ra)/2

    a1 = torch.zeros([z,r,r],device=sample.device)
    b1 = torch.zeros([z,r,r],device=sample.device)

    a1[:,0:ra,0:ca] = sample
    b1[:,0:rb,0:cb] = psf
    conv1 = ifft2(fft2(a1)*fft2(b1))
    projections = torch.abs(conv1[:,int(p1):int(r-p1),int(p1):int(r-p1)])
    if fp_res is not None and projections.shape[-1]!=fp_res: projections = upsample(projections,size=fp_res)
    return projections

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
    projections = torch.real(conv1[...,int(p1):int(r-p1),int(p1):int(r-p1)]).squeeze(0)
    if fp_res is not None and projections.shape[-1]!=fp_res: projections = upsample(projections,size=fp_res)
    return projections

def generate_fps_loop(psf,sample):
    if sample.ndim == 3: sample = sample.unsqueeze(0).unsqueeze(0)
    if sample.ndim == 4: sample = sample.unsqueeze(1)
    t,_,z,ra,ca = sample.shape
    u,_,rb,cb =psf.shape

    r = ra+rb-1
    p1 = (r-ra)/2

    a1 = torch.zeros([t,1,z,r,r],device=sample.device)
    b1 = torch.zeros([1,1,z,r,r],device=sample.device)

    a1[...,0:ra,0:ca] = sample
    projections = []
    for i in range(u):
        b1[...,0:rb,0:cb] = torch.unsqueeze(psf[i:i+1,...],0)
        conv1 = ifft2(fft2(a1)*fft2(b1))
        projections.append(torch.real(conv1[...,int(p1):int(r-p1),int(p1):int(r-p1)]))
    projections = torch.cat(projections,dim=1)
    return projections

def generate_fp_fft(psf_fft,sample):
    # same as DAOSLIMIT

    z,ra,ca = sample.shape
    _,r,_ =psf_fft.shape

    p1 = (r-ra)/2
    a1 = torch.zeros(r,r).to(sample.device)
    projections = []

    for i in range(z):
        a1[0:ra,0:ca] = sample[i,:,:]
        conv1 = ifft2(fft2(a1)*psf_fft[i,:,:])
        projections.append(torch.real(conv1[int(p1):int(r-p1),int(p1):int(r-p1)]))

    return torch.stack(projections,dim=0)

def normal(input):
    out = (input-torch.min(input))/(torch.max(input)-torch.min(input))
    return out

def load_uint16(path):
    input = tf.imread(path)
    input = input.astype(np.float32)
    out = torch.from_numpy(input)
    return out

def denoise(img):
    base_noise = torch.min(torch.mean(img,dim=0))
    img = img-base_noise
    print('Base Noise:',base_noise.item())
    img = torch.where(img>=0,img,torch.zeros_like(img))
    return img

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
    if not os.path.exists(save_dir): os.mkdir(save_dir)
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

def upsample_complex(img,factor=None,size=None):
    upreal = upsample(img.real,factor,size)
    upimag = upsample(img.imag,factor,size)
    up_img = torch.zeros([img.shape[0],upreal.shape[1],upreal.shape[2]],dtype=torch.cfloat)
    up_img.real = upreal;up_img.imag = upimag

    return up_img

def print_model_size(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.5fM' % (total / 1e6))

def save_temp(simulation,gt,save_dir,epoch,angle):
    diff = torch.abs(simulation-gt)
    temp = torch.stack([simulation,diff])
    temp_path = os.path.join(save_dir,save_dir+'_'+str(epoch)+'_'+str(angle)+'_diff.tif')
    save_results(temp,save_dir,temp_path) 

def log(writer,epoch,content=None,img=None):
    if content is not None: writer.add_scalar('Content Loss',content,epoch)
    if img is not None: writer.add_image('Center Slice',normal(img),epoch)

def psnr(original, reconstructed, max_value=1):

    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((original - reconstructed) ** 2)

    # Calculate PSNR
    psnr_value = 10 * torch.log10((max_value ** 2) / mse)

    return psnr_value

def load_psfs_1P(PSF_dir,Nnum,radius):
    uv_res = math.sqrt(Nnum)
    center_index = [(uv_res-1)/2,(uv_res-1)/2]
    postfix = os.listdir(PSF_dir)[0].split('.')[1]
    
    psfs = []
    for i in range(int(uv_res*uv_res)):
        index = [i%uv_res,int(i/uv_res)]
        distance = np.power(np.power(index[0]-center_index[0],2)+np.power(index[1]-center_index[1],2),0.5)
        if distance <= radius:
            psf_path = os.path.join(PSF_dir,'psf_'+str(i+1)+'.'+postfix)
            psf = torch.from_numpy(tf.imread(psf_path).astype(np.float32))
            psfs.append(psf)
            print('Load PSF:',psf_path,'Distance:',distance,'PSF max:',torch.max(psf).item())

    uv_num = len(psfs)
    print('PSFs Used:',uv_num)
    psfs = torch.stack(psfs,dim=0).squeeze()

    return psfs