from torch import optim, nn
from model import LF_PINE_c2f
from Utils.dataio import MicroData
from loss import fftloss,zloss,posloss
import torch.nn.functional as F
import time;import os
import Utils.utils as ut
import numpy as np
import torch
import math

########## Settings ##########
gpu_id = 7
log = ''

img_path = 'Projections/Fly_neurons_zoom4_2pSAM_z39_N13.tif'
psf_dir = 'PSF/PSF_zoom4_39dz1_128_N13'

psf_zoom = 4
img_zoom = 4
upfactor = 1
Nnum = 13

lr = 1e-3 # 5e-5 for realdata, 1e-4 for syndata
epochs = 100 # default 120
decay_step = 80 # default 100
coarse_epoch = 40

mpg_weight = 0 # range:0-10
fft_weight = 10 # range:0-10
ztv_weight = 10 # range:0-10
is_debackground = 0 # default 0
is_dynamic = 0 # default 0
##############################
projection_name = img_path.split('/')[-1]
curtime = time.strftime('%m_%d_%H_%M',time.localtime(time.time()))
notes = 'zTV%s_Up%s_FFT%s_MPG%s' %(str(ztv_weight),str(upfactor),str(fft_weight),str(mpg_weight))
if log != '' : notes = notes+'_'+log
project_name = curtime+'_LF-PINE_'+notes+'_'+projection_name.split('.tif')[0]
save_dir = os.path.join('./Results',project_name)
device = torch.device('cuda',gpu_id)

samplerate = psf_zoom/img_zoom
Dataset = MicroData(img_path,psf_dir,is_debackground,angle_norm=False,Nnum=Nnum)
hash_table_length,gt,gt_fft,psfs,amp = Dataset.getData()

hash_table_length = int(hash_table_length*(upfactor**2))
frames = gt.shape[0] if is_dynamic else 1
Nnum,z_res,psf_res,_ = psfs.shape
xy_res = hash_table_length/z_res
x_res = int(math.sqrt(xy_res))
fp_res = gt.shape[-1]
xguess_res = round(fp_res*samplerate)

render_limit = 180*512*512 # decided by GPU Memory
if hash_table_length%render_limit != 0 :render_times = int(hash_table_length/render_limit)+1 
else: render_times = int(hash_table_length/render_limit)
z_interval = math.ceil(z_res/render_times) if render_times>1 else z_res
coarse_zres = z_res if render_times==1 else int((z_res+1)/render_times)
if psf_res<512: psfs = psfs.to(device)
coarse_psfs = psfs[:,0:z_res:render_times,...]
print('Coarse Zres = %d, Render Times = %d'%(coarse_zres,render_times))

model = LF_PINE_c2f(x_res=x_res,coarse_zres=coarse_zres,z_res=z_res,in_features=3).to(device)
xguess = torch.zeros((coarse_zres,xguess_res,xguess_res),dtype=torch.float32).to(device)
raw = torch.zeros((coarse_zres,x_res,x_res),dtype=torch.float32).to(device)
optimizer = optim.Adam([{'params':model.parameters(), 'lr':lr}])

# training process
for frame in range(frames):
    if frame>0: epochs = 20;decay_step = int(epochs*0.8);coarse_epoch = epochs+1
    proj = gt[frame,...].to(device); Amp = Dataset.amp.to(device)
    for epoch in range(epochs):

        if epoch%decay_step==0: ut.Adjust_lr(optimizer,epoch,decay_step,0.3,lr)

        if epoch==coarse_epoch and frame==0: 
            with torch.no_grad(): model.coarse2fine()
            raw = ut.upsample(raw,dims=3,size=[z_res,x_res,x_res])
            xguess = ut.upsample(xguess,dims=3,size=[z_res,xguess_res,xguess_res])
            torch.cuda.empty_cache()

        is_fine = False if epoch<coarse_epoch and frame==0 else True

        for n in range(Nnum):
            angle = np.random.choice(np.arange(Nnum),size=1,replace=False)
            psf = psfs[angle,...] if is_fine else coarse_psfs[angle,...] # U,Z,X,Y
            if render_times>1: fp = ut.generate_fps(psf,xguess,fp_res)# U,Z,X,Y
            render_count = render_times if is_fine else 1
            for i in range(render_count):
                z_start = z_interval*i
                z_end = z_interval*(i+1) if z_interval*(i+1) <= z_res else z_res
                if not is_fine: z_start = 0; z_end = coarse_zres

                out = torch.squeeze(model(z_start,z_end,is_fine))
                raw[z_start:z_end,...] = out
                xguess[z_start:z_end,...] = ut.upsample(out,size=xguess_res) if xguess_res!=x_res else out
                if render_times>1: fp[:,z_start:z_end,...] = ut.generate_fps(psf[:,z_start:z_end,...],xguess[z_start:z_end,...],fp_res)
                else: fp = ut.generate_fps(psf,xguess,fp_res)
                simulation = torch.mean(fp,dim=-3)

                if mpg_weight>0: noisy_sim = ut.GenMixedNoisyLF(simulation,proj[angle,...])
                gt = proj[angle,...]

                loss_mse = F.mse_loss(simulation,gt)
                loss = loss_mse*1e3+posloss(xguess)*1e-2

                if ztv_weight>0: loss += zloss(raw)*ztv_weight
                if fft_weight>0: loss += fftloss(simulation,gt)*fft_weight
                if mpg_weight>0: loss += fftloss(noisy_sim.squeeze(0),gt)*mpg_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                xguess = xguess.detach()
                raw = raw.detach()
                if render_times>1:fp = fp.detach()

                print(f"[TRAIN] Frame:[{frame+1}/{frames}] Epoch:[{epoch+1}/{epochs}] Iter:[{n+1}/{Nnum}] Render Bunch:[{i+1}/{render_count}] MSE:{loss_mse.item()} Loss:{loss.item()}")

        if (epoch+1)%50==0 or (epoch+1)==epochs: ut.save_results(raw/Amp,save_dir,project_name+'_frame%.3d_epoch%d_raw.tif'%(frame,epoch)) 
        if (epoch+1)%10==0 and mpg_weight>0 : ut.save_results(simulation,save_dir,project_name+'_frame%.3d_fp.tif'%(frame)) 
        if (epoch+1)%10==0 and mpg_weight>0: ut.save_results(noisy_sim,save_dir,project_name+'_frame%.3d_nfp.tif'%(frame)) 
