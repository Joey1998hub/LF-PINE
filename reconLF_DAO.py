import math;import sys;import os;import time
from model import LF_PINE_DAO,ZernikeFitter
import torch;from torch import optim, nn
from loss import fftloss,zloss,posloss,xy_tv
from Utils.DAO_utils import genCircle,generate_psf_ab
from Utils.dataio import PhaseData
import numpy as np
import Utils.utils as ut
import tifffile as tf

gpu_id = 7
log = ''

img_path = 'Projections/USAF_RMS1.0_zoom4_2pSAM_z31_N13.tif'
psf_dir = 'PSF/PSFEx_zoom4_31dz1_N13'

psf_zoom = 4
img_zoom = 4
upfactor = 1
Nnum = 13

lr = 1e-3
epochs = 300 
decay_step = 200

ztv_weight = 0 
xytv_weight = 0 

polynum = 36
is_debackground = 0
# make directory
projection_name = img_path.split('/')[-1]
curtime = time.strftime('%m_%d_%H_%M',time.localtime(time.time()))
notes = 'zTV%d_Up%s_Zk%d' %(ztv_weight,str(upfactor),polynum)
project_name = curtime+'_LF-PINE-DAO_'+notes+'_Db%d_'%(is_debackground)+projection_name.split('.tif')[0]
save_dir = os.path.join('./Results',project_name)
device = torch.device('cuda',gpu_id)

samplerate = psf_zoom/img_zoom
Dataset = PhaseData(img_path,psf_dir,is_debackground)
hash_table_length,gt,gt_fft,psfs_fft = Dataset.getData()
hash_table_length = int(hash_table_length*(upfactor**2))
Nnum,z_res,psf_res,_ = psfs_fft.shape
xy_res = int(hash_table_length/z_res)
x_res = int(math.sqrt(xy_res))
fp_res = gt.shape[-1]
xguess_res = round(fp_res*samplerate)
render_limit = 180*512*512
if psf_res<512: psfs_fft = psfs_fft.to(device)
phase_mode = 1 if polynum>0 else 0

#####    Zernike Phase    #####
pupilsize = 1024; psfsize = 1024
phase_mask = genCircle(psf_res,int(psf_res/2)).to(device)
fplane = torch.zeros([psf_res,psf_res],dtype=torch.cfloat).to(device)
psf_max = torch.max(generate_psf_ab(psfs_fft[0,...].to(device),fplane,pupilsize,psfsize))

polymode = list(range(1,polynum)) if polynum>0 else [0]# polymode = [4]+list(range(6,15))
zernike = tf.imread('./PSF/Phase/zernike_kMax128_poly105.tif')
zernike = torch.from_numpy(zernike).float().to(device)
zernike  = zernike[polymode,...].unsqueeze(0)
zernike_fitter = ZernikeFitter(polymode,zernike,psf_res).to(device)
#####    Zernike Phase    #####

model = LF_PINE_DAO(hash_table_length=hash_table_length,in_features=3,z_res=z_res,out_features=1).to(device)
optimizer = optim.Adam([{'params':model.parameters(), 'lr':lr}, {'params':zernike_fitter.parameters(), 'lr':lr}])
criteon = nn.MSELoss()

xguess = torch.zeros((z_res,xguess_res,xguess_res),dtype=torch.float32).to(device)
raw = torch.zeros((z_res,x_res,x_res),dtype=torch.float32).to(device)
proj = gt[0,...].to(device); phases = []

if hash_table_length%render_limit != 0 :render_times = int(hash_table_length/render_limit)+1 
else: render_times = int(hash_table_length/render_limit)
z_interval = int(render_limit/xy_res) if render_times>1 else z_res

# training process
for epoch in range(epochs):
    if epoch%decay_step==0 : ut.Adjust_lr(optimizer,epoch,decay_step,0.5,lr)
    for angle in range(Nnum):
        psf_fft = psfs_fft[angle,...].to(device) if psf_res>=512 else psfs_fft[angle,...]
        for i in range(render_times):
            z_start = z_interval*i
            z_end = z_interval*(i+1) if z_interval*(i+1) <= z_res else z_res

            output = torch.squeeze(model(z_start,z_end).view(z_end-z_start,x_res,x_res,1))
            raw[z_start:z_end,...] = output
            xguess[z_start:z_end,...] = ut.upsample(output,size=xguess_res)

            # if phase_mode>0 and epoch>100: phase = zernike_fitter();fplane.imag = -phase*2*torch.pi
            if phase_mode>0: phase = zernike_fitter();fplane.imag = -phase*2*torch.pi
            psf_ab = generate_psf_ab(psf_fft,fplane,pupilsize,psfsize)

            if z_start == 0 : fp = ut.generate_fp(psf_ab,xguess,fp_res)
            else: fp[z_start:z_end,...] = ut.generate_fp(psf_ab[z_start:z_end,...],xguess[z_start:z_end,...],fp_res)
            simulation = torch.mean(fp,dim=0)

            loss_mse = criteon(simulation,proj[angle,...])
            loss_defocus = fftloss(simulation,proj[angle,...])    
            loss = loss_mse*1000+loss_defocus*10+posloss(xguess)*20

            if ztv_weight>0 : loss += zloss(raw)*ztv_weight
            if xytv_weight>0: loss += xy_tv(raw)*xytv_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            xguess.detach_()
            raw.detach_()
            fplane.detach_()
            if render_times>1:fp = fp.detach()

            print(f"[TRAIN] Frame:{1} Epoch:{epoch} Angle:{angle+1} Bunch:{i} Loss:{loss.item()}")

    if epoch%10==0 and phase_mode>0 and epoch>100: phases.append((phase*phase_mask).cpu())
    if (epoch+1)%10==0: ut.save_results(xguess.cpu()/Dataset.amp,save_dir,project_name+'.tif')

if phase_mode>0: ut.save_results(torch.stack(phases,dim=0),save_dir,project_name+'_estimated_wavefront.tif')
