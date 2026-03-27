import torch;import numpy as np
import torch.nn.functional as F
from torch.fft import fft2,fftshift
import tifffile as tf
import os;import sys
sys.path.append(os.getcwd())
import Utils.utils as ut

class MicroData():
    def __init__(self,image_path,psf_dir,is_debackground,angle_norm=False,Nnum=13):
        super().__init__()
        self.image = torch.from_numpy(tf.imread(image_path).astype(np.float32)).squeeze()
        if torch.max(self.image)>32767: self.image = self.image-32767 
        if self.image.ndim==3: self.image = self.image.unsqueeze(0)
        self.image = F.relu(self.image)

        if angle_norm: 
            for i in range(frames): self.image[i,...] = self.angleNorm(self.image[i,...])
        if is_debackground: self.image = self.debackground(self.image)

        self.amp = 1e-2/torch.mean(self.image)
        self.image = self.amp*self.image
        print('Image Amplification:',self.amp.item())

        psfs = ut.load_psfs(psf_dir,Nnum)
        
        self.psfs = psfs
        Nnum,z_res,psf_res,_ = psfs.shape
        frames,angles,fp_res,_ = self.image.shape
        if angles != Nnum: self.image = self.image.transpose(0,1)

        self.zxy = int(fp_res*fp_res*z_res)
        self.image_fft = fft2(self.image)
    
    def debackground(self,img):
        if img.ndim == 3: img = torch.unsqueeze(img,dim=0)
        t,u,x,y = img.shape
        for i in range(u):
            base_noise = ut.calBacknoise(img[:,i:i+1,...])
            img[:,i:i+1,...] = img[:,i:i+1,...]-base_noise
            print('Base Noise:',base_noise.item(),'Img Min:',torch.min(img[:,i:i+1,...]).item())
        img = F.relu(img)
        return img
    
    def angleNorm(self,img):
        img_mean = torch.mean(img)
        for i in range(img.shape[0]):
            img[i,...] = img[i,...]*(img_mean/torch.mean(img[i,...]))
        return img

    def getData(self):
        return self.zxy,self.image,self.image_fft,self.psfs,self.amp

class PhaseData():
    def __init__(self,image_path,psf_dir,is_debackground=False,angle_norm=False):
        super().__init__()
        self.image = torch.from_numpy(tf.imread(image_path).astype(np.float32)).squeeze()
        if torch.max(self.image)>32767: self.image = self.image-32767 
        if is_debackground: self.image = self.debackground(self.image)
        if self.image.ndim==3: self.image = self.image.unsqueeze(0)
        self.image = F.relu(self.image)

        # self.amp = 0.1/torch.mean(self.image)
        self.amp = torch.tensor([1]).to(self.image.device)
        self.image = self.amp*self.image
        print('Image Amplification:',self.amp.item())

        self.psfs_fft = self.load_psfExs(psf_dir)
        Nnum,z_res,psf_res,_ = self.psfs_fft.shape
        frames,_,fp_res,_ = self.image.shape

        if angle_norm: 
            for i in range(frames): self.image[i,...] = self.angleNorm(self.image[i,...])

        self.zxy = int(fp_res*fp_res*z_res)
        self.image_fft = fft2(self.image)
    
    def load_psfExs(self,PSF_dir):

        psfs_fft = []
        postfix = os.listdir(PSF_dir)[0].split('.')[1]
        Nnum = int(len(os.listdir(PSF_dir))/2)
        for index in range(Nnum):
            psf_real_path = os.path.join(PSF_dir,'psf_'+str(index+1)+'_real.'+postfix)
            psf_imag_path = os.path.join(PSF_dir,'psf_'+str(index+1)+'_imag.'+postfix)
            psf_real = torch.from_numpy(tf.imread(psf_real_path).astype(np.float32))
            psf_imag = torch.from_numpy(tf.imread(psf_imag_path).astype(np.float32))
            psf_fft = torch.zeros([psf_real.shape[0],psf_real.shape[1],psf_real.shape[2]],dtype=torch.cfloat)
            psf_fft.real=psf_real ; psf_fft.imag=psf_imag
            psfs_fft.append(psf_fft)
            print('Load PSFEx:',psf_real_path,'PSF max:',torch.max(psf_real).item())
            print('Load PSFEx:',psf_imag_path,'PSF max:',torch.max(psf_imag).item())

        psfs_fft = torch.stack(psfs_fft,dim=0).squeeze()
        return psfs_fft
    
    def angleNorm(self,img):
        img_mean = torch.mean(img)
        for i in range(img.shape[0]):
            img[i,...] = img[i,...]*(img_mean/torch.mean(img[i,...]))
        return img

    def getData(self):
        return self.zxy, self.image, self.image_fft ,self.psfs_fft
