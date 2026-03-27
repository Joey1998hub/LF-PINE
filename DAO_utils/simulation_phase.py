import torch
import tifffile as tf
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
from Utils.utils import generate_fp,normal

RMS = 0.1
PSF_dir = 'PSF/PSF_zoom4_31dz1_N13_1024_Phase_PSFRes786_Polynum36_RMS%s'%str(RMS)
sample_path = 'Samples/USAF31.tif'
is_norm = 0
Nnum = 13

Projection_dir = './Projections/Phase'
sample_name = sample_path.split('/')[-1]

sample = tf.imread(sample_path).astype(np.float32)
sample = torch.from_numpy(sample).squeeze()
if torch.max(sample)>32767: sample = sample-32767 

if is_norm==1: 
    sample = normal(sample)
    tf.imwrite(sample_path[:-4]+'_normal.tif',sample.cpu().numpy())
print('Sample Max:',torch.max(sample).numpy())

if sample.ndim==3:sample = sample.unsqueeze(0)
frames,z_res,x_res,y_res = sample.shape
projections=torch.zeros(size=[frames,Nnum,x_res,y_res])

psfs = []
for i in range(Nnum):
    psf_name = 'psf_'+str(i+1)+'.tiff'
    psf_path = os.path.join(PSF_dir,psf_name)
    psf = torch.from_numpy(tf.imread(psf_path).astype(np.float32)).squeeze()
    print('PSF Max:',torch.max(psf).cpu().numpy())
    psfs.append(psf)

for k in range(frames):
    for i in range(Nnum):
        fp = generate_fp(psfs[i],sample[k,...])
        projection = torch.mean(fp,dim=0)
        print('Projection[%d,%d] Mean:'%(k,i),torch.mean(projection).cpu().numpy())
        projections[k,i,:,:] = projection

projection_path = os.path.join(Projection_dir,sample_name[:-4]+'_fp_SE_RMS%s.tif'%(str(RMS)))
tf.imwrite(projection_path,projections.cpu().numpy())
print('Saved:',projection_path)