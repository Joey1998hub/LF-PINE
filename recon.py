"""
LF-PINE: unified light-field 3D reconstruction (static + dynamic).

One script reconstructs light-field (LF) measurements into 3D volumes for every
released dataset. All dataset-specific parameters live in JSON files under
./settings:

    python recon.py --config settings/highna_fp.json
    python recon.py --config settings/bpae_cell.json
    python recon.py --config settings/slfm_selectR4.json
    python recon.py --config settings/phase_usaf31.json

Three reconstruction pipelines are selected by the "mode" and "is_dynamic"
config keys:
  - mode="intensity", is_dynamic=false: single-LF intensity reconstruction
    with a hash-table volume and a coarse-to-fine schedule (BPAE_cell,
    HighNA10, sLFM_data_selectR4).
  - mode="phase", is_dynamic=false: joint intensity + aberrated-phase
    (Zernike) reconstruction from complex pupil PSFs (USAF_phase).
  - mode="intensity", is_dynamic=true: dynamic time-series reconstruction.
    The first frame is reconstructed from scratch ("epochs"); every following
    frame warm-starts from the previous volume with a partially re-randomized
    hash table ("reset_ratio") and fewer epochs ("dynamic_epochs"), using an
    FFT-cached forward model. Enable it from any intensity config with
    --set is_dynamic=true and a multi-frame img_path.

Any config key can be overridden from the command line, which makes batch
processing of several measurements easy, e.g.:

    python recon.py --config settings/highna_fp.json --gpu_id 1 \
        --set img_path=Projections/HighNA10/1_z61.tif \
        --set gt_path=Samples/highNA_10/1_z61.tif

Run from the repository root so that relative paths in the JSON files resolve.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import tifffile as tf
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch import optim

import Utils.utils as ut
from loss import fftloss, nllp_loss, posloss, zloss
from model import HashMLP, HashVolume, HashVolume_c2f, ZernikeFitter
from Utils.dataio import MicroData, PhaseData
from Utils.phase_utils import genCircle, generate_psf_ab

########## Config ##########
DEFAULTS = dict(
    # general
    name='lf_recon',
    mode='intensity',        # 'intensity' | 'phase'
    is_dynamic=False,        # False: reconstruct frame_index; True: reconstruct a frame range
    img_path='',
    psf_dir='',
    gt_path='',              # optional ground-truth volume for PSNR/SSIM evaluation
    # optics / geometry
    psf_zoom=4,
    img_zoom=4,
    upfactor=1,
    Nnum=13,
    z_sample=0,              # >0: keep only the central z_sample PSF planes
    frame_index=0,           # static mode: which frame of a multi-frame LF stack to reconstruct
    # dynamic mode (is_dynamic=true)
    start_frame=0,
    end_frame=None,          # None: reconstruct to the last frame (exclusive)
    dynamic_epochs=None,   # None: epochs//10; epochs for every frame after the first one
    dynamic_fft_weight=None, # None: 4*fft_weight
    reset_ratio=0.2,         # fraction of hash-table entries re-randomized per frame
    N_sample=1,              # number of angles sampled per iteration
    # optimization
    lr=1e-3,
    epochs=100,              # static mode: total epochs; dynamic mode: epochs of the first frame
    coarse_epoch=40,         # static intensity mode: switch from coarse to fine table
    iters=100,
    lr_decay=0.3,
    decay_ratio=0.8,         # decay lr every int(epochs*decay_ratio)
    mse_weight=1e3,
    pos_weight=1e-2,
    fft_weight=10,
    mpg_weight=0,            # mixed Poisson-Gaussian FFT loss for low-SNR data (L_MPG-FFT in the paper)
    ztv_weight=0,
    nllp_weight=0,
    cutoff_mode=0,           # 0: no cutoff; 1: rough; 2: precise (resolution-matched blur)
    cutoff_sigma=0,          # >0: override the cutoff sigma directly
    is_debackground=0,
    angle_norm=False,
    sim_mode='sum',          # 'sum' | 'mean' over the z axis of the forward projection
    render_limit=200*512*512,
    save_normalized=True,    # save raw/amp (physical scale) instead of raw network output
    eval=True,               # evaluate against gt_path when it is given
    save_div=10,             # save/evaluate every epochs//save_div epochs
    # phase mode only
    polynum=38,              # fit Zernike modes 1..polynum-1 (the first 37, piston excluded); 0 disables phase retrieval
    zernike_path='PSF/Phase/zernike_kMax128_poly105.tif',
    pupilsize=1024,
    psfsize=360,
    psf_sim_amp=1e8,         # amp used when the simulation PSFs were generated
    phase_z_layers=0,        # <=0: update aberrated PSFs on all z planes
    dao_init=0,              # start phase retrieval after this epoch
    zernike_lr_ratio=0.1,
    # misc
    gpu_id=0,
    tag='',
    seed=None,
    save_root='./Results',
)


class Config(dict):
    """dict with attribute access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def parse_args():
    parser = argparse.ArgumentParser(description='LF-PINE unified LF reconstruction (static + dynamic)')
    parser.add_argument('--config', type=str, required=True, help='path to a settings JSON file')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--psf_dir', type=str, default=None)
    parser.add_argument('--gt_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        help='override any config key: --set key=value (value parsed as JSON when possible)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = Config(json.load(f))
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    for key in ('gpu_id', 'img_path', 'psf_dir', 'gt_path', 'epochs', 'lr', 'tag', 'save_root'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    for item in args.overrides:
        key, _, value = item.partition('=')
        try:
            value = json.loads(value)
        except ValueError:
            pass
        cfg[key] = value

    if cfg.dynamic_fft_weight is None:
        cfg.dynamic_fft_weight = cfg.fft_weight*4
    if cfg.dynamic_epochs is None:
        cfg.dynamic_epochs = max(1, cfg.epochs//10)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    return cfg


########## Shared helpers ##########
def make_save_dir(cfg, notes):
    projection_name = os.path.basename(cfg.img_path).split('.tif')[0]
    curtime = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
    project_name = curtime + '_LFP_' + notes + '_' + projection_name
    if cfg.tag != '':
        project_name = project_name + '_' + cfg.tag
    save_dir = os.path.join(cfg.save_root, project_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(dict(cfg), f, indent=2)
    print('Save to:', save_dir)
    return save_dir, projection_name


def get_cutoff_blur(cfg):
    """Gaussian blur applied to the network output, matching the reconstruction
    resolution to the measured data (cutoff)."""
    sigma = 0
    kernel_size = 3
    if cfg.cutoff_mode == 1:
        sigma = 0.75
        kernel_size = 3
    elif cfg.cutoff_mode == 2:
        sigma = (0.3/((688/cfg.img_zoom)/(512*cfg.upfactor)))/2.355
        kernel_size = int(2*sigma)*2+1
    if cfg.cutoff_sigma > 0:
        sigma = cfg.cutoff_sigma
        kernel_size = int(2*sigma)*2+1
    return sigma, kernel_size


def load_gt_volume(cfg):
    gt_vol = torch.from_numpy(tf.imread(cfg.gt_path).astype(np.float32)).relu().squeeze()
    if gt_vol.ndim == 2:
        gt_vol = gt_vol.unsqueeze(0)
    print('Load GT:', cfg.gt_path, 'shape:', tuple(gt_vol.shape))
    return gt_vol


def cal_metrics(recon, gt_vol):
    """PSNR / SSIM between a reconstructed volume and the GT volume.
    Both are relu-ed and normalized by the GT max; the reconstruction is
    resized in xy when the lateral resolutions differ."""
    recon = recon.detach().cpu().float().relu().squeeze()
    if recon.ndim == 2:
        recon = recon.unsqueeze(0)
    if recon.shape[-1] != gt_vol.shape[-1]:
        recon = ut.upsample(recon, dims=2, size=gt_vol.shape[-1], mode='bicubic')
    sample_max = torch.max(gt_vol)
    gt_n = gt_vol/sample_max
    recon_n = recon/sample_max
    mse = torch.mean((recon_n-gt_n)**2)
    psnr_value = (10*torch.log10(1.0/mse)).item()
    ssim_value = float('nan')
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim_value = ssim(recon_n.unsqueeze(0), gt_n.unsqueeze(0)).item()
    except ImportError:
        print('torchmetrics not installed, skip SSIM')
    return psnr_value, ssim_value


########## Static intensity mode ##########
def run_intensity(cfg):
    device = torch.device('cuda', cfg.gpu_id)
    notes = 'zTV%s_Up%s_FFT%s_Nlp%s_Db%d_Cut%d_%s' % (
        str(cfg.ztv_weight), str(cfg.upfactor), str(cfg.fft_weight),
        str(cfg.nllp_weight), cfg.is_debackground, cfg.cutoff_mode, cfg.sim_mode)
    if cfg.mpg_weight > 0:
        notes += '_MPG%s' % str(cfg.mpg_weight)
    save_dir, projection_name = make_save_dir(cfg, notes)

    samplerate = cfg.psf_zoom/cfg.img_zoom
    Dataset = MicroData(cfg.img_path, cfg.psf_dir, cfg.is_debackground,
                        angle_norm=cfg.angle_norm, Nnum=cfg.Nnum, z_sample=cfg.z_sample)
    hash_table_length, gt, psfs, amp = Dataset.getData()

    hash_table_length = int(hash_table_length*(cfg.upfactor**2))
    Nnum, z_res, psf_res, _ = psfs.shape
    xy_res = hash_table_length/z_res
    x_res = int(math.sqrt(xy_res))
    fp_res = gt.shape[-1]
    xguess_res = round(fp_res*samplerate)

    sigma, kernel_size = get_cutoff_blur(cfg)
    sample_mode = 'area' if x_res > xguess_res else 'bicubic'

    render_limit = cfg.render_limit
    iters = cfg.iters
    if hash_table_length % render_limit != 0:
        render_times = int(hash_table_length/render_limit)+1
    else:
        render_times = int(hash_table_length/render_limit)
    z_interval = math.ceil(z_res/render_times) if render_times > 1 else z_res
    coarse_zres = z_res if render_times == 1 else int((z_res+1)/render_times)
    if psf_res < 512:
        psfs = psfs.to(device)
    coarse_psfs = psfs[:, 0:z_res:render_times, ...]
    print('Coarse Zres = %d, Render Times = %d' % (coarse_zres, render_times))
    print('cutoff_mode = %d, sigma = %s, sim_mode = %s' % (cfg.cutoff_mode, str(sigma), cfg.sim_mode))

    model = HashVolume_c2f(x_res=x_res, coarse_zres=coarse_zres, z_res=z_res, in_features=3).to(device)
    xguess = torch.zeros((coarse_zres, xguess_res, xguess_res), dtype=torch.float32).to(device)
    raw = torch.zeros((coarse_zres, x_res, x_res), dtype=torch.float32).to(device)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': cfg.lr}])

    epochs = cfg.epochs
    coarse_epoch = cfg.coarse_epoch
    decay_step = max(1, int(epochs*cfg.decay_ratio))
    save_every = max(1, epochs//cfg.save_div)

    proj = gt[cfg.frame_index, ...].to(device)
    Amp = Dataset.amp.to(device)
    gt_vol = load_gt_volume(cfg) if (cfg.eval and cfg.gt_path) else None
    best_psnr = -1

    # training process
    for epoch in range(epochs):
        if epoch % decay_step == 0:
            ut.Adjust_lr(optimizer, epoch, decay_step, cfg.lr_decay, cfg.lr)

        if epoch == coarse_epoch:
            with torch.no_grad():
                model.coarse2fine()
            raw = ut.upsample(raw, dims=3, size=[z_res, x_res, x_res])
            xguess = ut.upsample(xguess, dims=3, size=[z_res, xguess_res, xguess_res])
            torch.cuda.empty_cache()

        is_fine = False if epoch < coarse_epoch else True
        start_time = time.time()

        for n in range(iters):
            angle = np.random.choice(np.arange(Nnum), size=1, replace=False)
            lf = proj[angle, ...]
            psf = psfs[angle, ...] if is_fine else coarse_psfs[angle, ...]  # U,Z,X,Y
            if render_times > 1:
                fp = ut.generate_fps(psf, xguess, fp_res)  # U,Z,X,Y
            render_count = render_times if is_fine else 1
            for i in range(render_count):
                z_start = z_interval*i
                z_end = z_interval*(i+1) if z_interval*(i+1) <= z_res else z_res
                if not is_fine:
                    z_start = 0
                    z_end = coarse_zres

                out = torch.squeeze(model(z_start, z_end, is_fine))
                if sigma > 0:
                    out = gaussian_blur2d(out.unsqueeze(0), kernel_size=kernel_size, sigma=(sigma, sigma)).squeeze(0)
                raw[z_start:z_end, ...] = out
                xguess[z_start:z_end, ...] = ut.upsample(out, size=xguess_res, mode=sample_mode) if xguess_res != x_res else out
                if render_times > 1:
                    fp[:, z_start:z_end, ...] = ut.generate_fps(psf[:, z_start:z_end, ...], xguess[z_start:z_end, ...], fp_res)
                else:
                    fp = ut.generate_fps(psf, xguess, fp_res)
                simulation = torch.sum(fp, dim=-3) if cfg.sim_mode == 'sum' else torch.mean(fp, dim=-3)

                loss_mse = F.mse_loss(simulation, lf)
                loss = loss_mse*cfg.mse_weight+posloss(xguess)*cfg.pos_weight

                if cfg.ztv_weight > 0:
                    loss += zloss(raw)*cfg.ztv_weight
                if cfg.fft_weight > 0:
                    loss += fftloss(simulation, lf)*cfg.fft_weight
                if cfg.mpg_weight > 0:
                    noisy_sim = ut.GenMixedNoisyLF(simulation, lf)
                    loss += fftloss(noisy_sim.squeeze(0), lf)*cfg.mpg_weight
                if cfg.nllp_weight > 0:
                    loss += nllp_loss(simulation, lf)*cfg.nllp_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                xguess = xguess.detach()
                raw = raw.detach()
                if render_times > 1:
                    fp = fp.detach()

                speed = (time.time()-start_time)/((n+1)*render_count)
                print(f"[TRAIN] Epoch:[{epoch+1}/{epochs}] Iter:[{n+1}/{iters}] Render Bunch:[{i+1}/{render_count}] MSE:{loss_mse.item():.5g} Loss:{loss.item():.5g} Speed:{speed:.3g} s/it")

        if (epoch+1) % save_every == 0 or (epoch+1) == epochs:
            vol = raw/Amp if cfg.save_normalized else raw
            ut.save_results(vol, save_dir, projection_name+'_epoch%03d.tif' % (epoch+1))
            if gt_vol is not None:
                psnr_value, ssim_value = cal_metrics(raw/Amp, gt_vol)
                msg = '[EVAL] Epoch:[%d/%d] PSNR: %.4f dB SSIM: %.4f (best: %.4f dB)' % (
                    epoch+1, epochs, psnr_value, ssim_value, max(best_psnr, psnr_value))
                print(msg)
                with open(os.path.join(save_dir, 'eval_log.txt'), 'a') as f:
                    f.write(msg+'\n')
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    ut.save_results(raw/Amp, save_dir, 'best.tif')

    vol = raw/Amp if cfg.save_normalized else raw
    ut.save_results(vol, save_dir, projection_name+'_recon.tif')

    if gt_vol is not None:
        psnr_value, ssim_value = cal_metrics(raw/Amp, gt_vol)
        msg = '[FINAL] PSNR: %.4f dB SSIM: %.4f | best PSNR: %.4f dB' % (psnr_value, ssim_value, best_psnr)
        print(msg)
        with open(os.path.join(save_dir, 'eval.txt'), 'w') as f:
            f.write(json.dumps(dict(cfg), indent=2)+'\n')
            f.write(msg+'\n')


########## Phase mode ##########
def run_phase(cfg):
    device = torch.device('cuda', cfg.gpu_id)
    notes = 'zTV%s_Up%s_Zk%d_Pz%d_FFT%s_Nlp%s_Cut%d_Db%d' % (
        str(cfg.ztv_weight), str(cfg.upfactor), cfg.polynum, cfg.phase_z_layers,
        str(cfg.fft_weight), str(cfg.nllp_weight), cfg.cutoff_mode, cfg.is_debackground)
    save_dir, projection_name = make_save_dir(cfg, notes)

    Dataset = PhaseData(cfg.img_path, cfg.psf_dir, cfg.is_debackground,
                        angle_norm=cfg.angle_norm, z_sample=cfg.z_sample)
    hash_table_length, gt, psfs_fft = Dataset.getData()
    hash_table_length = int(hash_table_length*(cfg.upfactor**2))
    Nnum, z_res, psf_res, _ = psfs_fft.shape
    phase_z_layers = z_res if cfg.phase_z_layers <= 0 else min(cfg.phase_z_layers, z_res)
    phase_z_start = int((z_res-phase_z_layers)//2)
    phase_z_end = phase_z_start+phase_z_layers
    xy_res = int(hash_table_length/z_res)
    x_res = int(math.sqrt(xy_res))
    fp_res = gt.shape[-1]
    xguess_res = round(fp_res*(cfg.psf_zoom/cfg.img_zoom))
    render_limit = cfg.render_limit
    psfs_fft = psfs_fft.to(device)
    phase_mode = 1 if cfg.polynum > 0 else 0
    iters = cfg.iters

    #####    Zernike Phase    #####
    phase_mask = genCircle(psf_res, int(psf_res/2)).to(device)
    fplane = torch.zeros([psf_res, psf_res], dtype=torch.cfloat).to(device)
    psf_max = torch.max(generate_psf_ab(psfs_fft[0:1, ...].to(device), fplane, cfg.pupilsize, cfg.psfsize, amp=cfg.psf_sim_amp))
    psf_amp = 1e-3/psf_max
    # physical output scale: data fits amp*proj = conv(xguess, psf_amp*psf) while
    # proj = conv(GT, psf), so GT = xguess*psf_amp/amp
    output_scale = (psf_amp/Dataset.amp).item()

    polymode = list(range(1, cfg.polynum)) if cfg.polynum > 0 else [0]  # polymode = [4]+list(range(6,15))
    zernike = tf.imread(cfg.zernike_path)
    zernike = torch.from_numpy(zernike).float().to(device)
    zernike = zernike[polymode, ...].unsqueeze(0)
    zernike_fitter = ZernikeFitter(polymode, zernike, psf_res).to(device)
    #####    Zernike Phase    #####

    model = HashMLP(hash_table_length=hash_table_length, in_features=3, z_res=z_res, out_features=1).to(device)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': cfg.lr},
                            {'params': zernike_fitter.parameters(), 'lr': cfg.lr*cfg.zernike_lr_ratio}])

    xguess = torch.zeros((z_res, xguess_res, xguess_res), dtype=torch.float32).to(device)
    raw = torch.zeros((z_res, x_res, x_res), dtype=torch.float32).to(device)
    proj = gt[cfg.frame_index, ...].to(device)
    phases = []

    sigma, kernel_size = get_cutoff_blur(cfg)
    sample_mode = 'area' if x_res > xguess_res else 'bicubic'

    if hash_table_length % render_limit != 0:
        render_times = int(hash_table_length/render_limit)+1
    else:
        render_times = int(hash_table_length/render_limit)
    z_interval = int(render_limit/xy_res) if render_times > 1 else z_res

    psfs_ideal = generate_psf_ab(psfs_fft, fplane, cfg.pupilsize, cfg.psfsize, amp=cfg.psf_sim_amp)*psf_amp
    print('Phase PSF update z range: [%d, %d], layers=%d/%d' % (phase_z_start, phase_z_end, phase_z_layers, z_res))

    epochs = cfg.epochs
    decay_step = max(1, int(epochs*cfg.decay_ratio))
    save_every = max(1, epochs//cfg.save_div)
    gt_vol = load_gt_volume(cfg) if (cfg.eval and cfg.gt_path) else None
    best_psnr = -1

    # training process
    for epoch in range(epochs):
        if epoch % decay_step == 0:
            ut.Adjust_lr(optimizer, epoch, decay_step, cfg.lr_decay, cfg.lr)
        for n in range(iters):
            angle = np.random.choice(np.arange(Nnum), size=1, replace=False)
            psf_fft = psfs_fft[angle, ...]
            for i in range(render_times):
                z_start = z_interval*i
                z_end = z_interval*(i+1) if z_interval*(i+1) <= z_res else z_res

                output = torch.squeeze(model(z_start, z_end).view(z_end-z_start, x_res, x_res, 1))
                if sigma > 0:
                    output = gaussian_blur2d(output.unsqueeze(0), kernel_size=kernel_size, sigma=(sigma, sigma)).squeeze(0)
                raw[z_start:z_end, ...] = output
                xguess[z_start:z_end, ...] = ut.upsample(output, size=xguess_res, mode=sample_mode)

                if phase_mode > 0 and epoch > cfg.dao_init:
                    phase = zernike_fitter()
                    fplane.imag = -phase*2*torch.pi
                    psf_ab = psfs_ideal[angle, ...].clone()
                    psf_ab[:, phase_z_start:phase_z_end, ...] = generate_psf_ab(
                        psf_fft[:, phase_z_start:phase_z_end, ...], fplane, cfg.pupilsize, cfg.psfsize, amp=cfg.psf_sim_amp)*psf_amp
                else:
                    psf_ab = psfs_ideal[angle, ...]

                if z_start == 0:
                    fp = ut.generate_fps(psf_ab, xguess, fp_res)
                else:
                    fp[:, z_start:z_end, ...] = ut.generate_fps(psf_ab[:, z_start:z_end, ...], xguess[z_start:z_end, ...], fp_res)
                simulation = torch.mean(fp, dim=-3)

                lf = proj[angle, ...]
                loss_mse = F.mse_loss(simulation, lf)
                loss = loss_mse*cfg.mse_weight+posloss(xguess)*cfg.pos_weight
                if cfg.fft_weight > 0:
                    loss += fftloss(simulation, lf)*cfg.fft_weight
                if cfg.ztv_weight > 0:
                    loss += zloss(raw)*cfg.ztv_weight
                if cfg.nllp_weight > 0:
                    loss += nllp_loss(simulation, lf)*cfg.nllp_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                xguess.detach_()
                raw.detach_()
                fplane.detach_()
                if render_times > 1:
                    fp = fp.detach()

                print(f"[TRAIN] Epoch:[{epoch+1}/{epochs}] Render Bunch:[{i+1}/{render_times}] Iter:[{n+1}/{iters}] MSE:{loss_mse.item():.5g} Loss:{loss.item():.5g}")

        if epoch % 10 == 0 and phase_mode > 0 and epoch > cfg.dao_init:
            phases.append((phase*phase_mask).cpu())
        if (epoch+1) % save_every == 0 or (epoch+1) == epochs:
            ut.save_results(raw.cpu()*output_scale, save_dir, projection_name+'_epoch%03d.tif' % (epoch+1))
            if gt_vol is not None:
                psnr_value, ssim_value = cal_metrics(raw*output_scale, gt_vol)
                msg = '[EVAL] Epoch:[%d/%d] PSNR: %.4f dB SSIM: %.4f (best: %.4f dB)' % (
                    epoch+1, epochs, psnr_value, ssim_value, max(best_psnr, psnr_value))
                print(msg)
                with open(os.path.join(save_dir, 'eval_log.txt'), 'a') as f:
                    f.write(msg+'\n')
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    ut.save_results(raw.cpu()*output_scale, save_dir, 'best.tif')

    ut.save_results(raw.cpu()*output_scale, save_dir, projection_name+'_recon.tif')

    if phase_mode > 0:
        np.savetxt(os.path.join(save_dir, 'zernike_coef.txt'),
                   torch.squeeze(zernike_fitter.zernike_coef).detach().cpu().numpy(), fmt='%.6f')
        if len(phases) > 0:
            ut.save_results(torch.stack(phases, dim=0), save_dir, projection_name+'_phase.tif')

    if gt_vol is not None:
        psnr_value, ssim_value = cal_metrics(raw*output_scale, gt_vol)
        msg = '[FINAL] PSNR: %.4f dB SSIM: %.4f | best PSNR: %.4f dB' % (psnr_value, ssim_value, best_psnr)
        print(msg)
        with open(os.path.join(save_dir, 'eval.txt'), 'w') as f:
            f.write(json.dumps(dict(cfg), indent=2)+'\n')
            f.write(msg+'\n')


########## Dynamic mode (is_dynamic=true) ##########
def run_dynamic(cfg):
    device = torch.device('cuda', cfg.gpu_id)
    frame_range_note = 'frames%d-%s' % (cfg.start_frame, 'end' if cfg.end_frame is None else cfg.end_frame)
    notes = 'zTV%s_Up%s_FFT%s_Nlp%s_Db%d_Cut%d_%s' % (
        str(cfg.ztv_weight), str(cfg.upfactor), str(cfg.fft_weight),
        str(cfg.nllp_weight), cfg.is_debackground, cfg.cutoff_mode, frame_range_note)
    if cfg.mpg_weight > 0:
        notes += '_MPG%s' % str(cfg.mpg_weight)
    save_dir, projection_name = make_save_dir(cfg, notes)

    samplerate = cfg.psf_zoom/cfg.img_zoom
    Dataset = MicroData(cfg.img_path, cfg.psf_dir, cfg.is_debackground,
                        angle_norm=cfg.angle_norm, Nnum=cfg.Nnum, z_sample=cfg.z_sample)
    hash_table_length, gt, psfs, amp = Dataset.getData()

    total_frames = gt.shape[0]
    end_frame = total_frames if cfg.end_frame is None else cfg.end_frame
    frame_indices = list(range(cfg.start_frame, end_frame))
    gt = gt[cfg.start_frame:end_frame, ...]
    print('Reconstruct frames: %d-%d (exclusive end), total selected: %d / %d'
          % (cfg.start_frame, end_frame, len(frame_indices), total_frames))

    hash_table_length = int(hash_table_length*(cfg.upfactor**2))
    frames = gt.shape[0]
    fp_res = gt.shape[-1]
    Nnum, z_res, psf_res, _ = psfs.shape
    xy_res = hash_table_length/z_res
    x_res = int(math.sqrt(xy_res))
    xguess_res = round(fp_res*samplerate)

    sigma, kernel_size = get_cutoff_blur(cfg)
    sample_mode = 'area' if x_res > xguess_res else 'bicubic'

    render_limit = cfg.render_limit
    iters = cfg.iters
    if hash_table_length % render_limit != 0:
        render_times = int(hash_table_length/render_limit)+1
    else:
        render_times = int(hash_table_length/render_limit)
    z_interval = math.ceil(z_res/render_times) if render_times > 1 else z_res
    psfs = psfs.to(device)
    print('Zres = %d, Render Times = %d' % (z_res, render_times))

    model = HashVolume(x_res=x_res, z_res=z_res, in_features=3).to(device)
    xguess = torch.zeros((z_res, xguess_res, xguess_res), dtype=torch.float32).to(device)
    raw = torch.zeros((z_res, x_res, x_res), dtype=torch.float32).to(device)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': cfg.lr}])

    psfs_fft, fft_res = ut.generate_psfs_fft(psfs, xguess_res)
    print('Cached PSF FFT:', psfs_fft.shape)

    sample_buffer = torch.zeros((1, 1, z_res, fft_res, fft_res), device=device, dtype=torch.float32)
    if render_times > 1:
        sample_block_buffer = torch.zeros((1, 1, z_interval, fft_res, fft_res), device=device, dtype=torch.float32)

    # training process
    for frame in range(frames):
        raw_frame = frame_indices[frame]
        if frame > 0:
            # warm start: fewer epochs, stronger fft prior, partially re-randomized table
            epochs = cfg.dynamic_epochs
            fft_weight = cfg.dynamic_fft_weight
            model.partial_restart_table(cfg.reset_ratio)
            print(f'Hashtable partial restart, reset_ratio={cfg.reset_ratio}')
            optimizer = optim.Adam([{'params': model.parameters(), 'lr': cfg.lr}])
        else:
            epochs = cfg.epochs
            fft_weight = cfg.fft_weight
        decay_step = max(1, int(epochs*cfg.decay_ratio))

        proj = gt[frame, ...].to(device)
        for epoch in range(epochs):
            if epoch % decay_step == 0:
                ut.Adjust_lr(optimizer, epoch, decay_step, cfg.lr_decay, cfg.lr)
            start_time = time.time()
            for n in range(iters):
                angle = np.random.choice(np.arange(Nnum), size=cfg.N_sample, replace=False)
                lf = proj[angle, ...]
                psf_fft_buffer = psfs_fft[angle, ...].unsqueeze(0)
                render_count = render_times

                if render_times > 1:
                    sample_buffer.zero_()
                    sample_buffer[..., 0:xguess_res, 0:xguess_res] = xguess.unsqueeze(0).unsqueeze(0)
                    fp = ut.generate_fps_fft(sample_buffer, psf_fft_buffer, xguess_res, fp_res)  # U,Z,X,Y

                for i in range(render_count):
                    z_start = z_interval*i
                    z_end = z_interval*(i+1) if z_interval*(i+1) <= z_res else z_res
                    z_count = z_end-z_start

                    out = torch.squeeze(model(z_start, z_end))
                    if sigma > 0:
                        out = gaussian_blur2d(out.unsqueeze(0), kernel_size=kernel_size, sigma=(sigma, sigma)).squeeze(0)
                    raw[z_start:z_end, ...] = out
                    xguess[z_start:z_end, ...] = ut.upsample(out, size=xguess_res, mode=sample_mode) if xguess_res != x_res else out

                    if render_times > 1:
                        cur_sample_buffer = sample_block_buffer[:, :, 0:z_count, ...]
                        cur_sample_buffer.zero_()
                        cur_sample_buffer[..., 0:xguess_res, 0:xguess_res] = xguess[z_start:z_end, ...].unsqueeze(0).unsqueeze(0)
                        fp[:, z_start:z_end, ...] = ut.generate_fps_fft(cur_sample_buffer, psf_fft_buffer[:, :, z_start:z_end, ...], xguess_res, fp_res)
                    else:
                        sample_buffer.zero_()
                        sample_buffer[..., 0:xguess_res, 0:xguess_res] = xguess.unsqueeze(0).unsqueeze(0)
                        fp = ut.generate_fps_fft(sample_buffer, psf_fft_buffer, xguess_res, fp_res)

                    simulation = torch.mean(fp, dim=-3)

                    loss_mse = F.mse_loss(simulation, lf)
                    loss = loss_mse*cfg.mse_weight+posloss(xguess)*cfg.pos_weight

                    if cfg.ztv_weight > 0:
                        loss += zloss(raw)*cfg.ztv_weight
                    if fft_weight > 0:
                        loss += fftloss(simulation, lf)*fft_weight
                    if cfg.mpg_weight > 0:
                        noisy_sim = ut.GenMixedNoisyLF(simulation, lf)
                        loss += fftloss(noisy_sim.squeeze(0), lf)*cfg.mpg_weight
                    if cfg.nllp_weight > 0:
                        loss += nllp_loss(simulation, lf)*cfg.nllp_weight

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    sample_buffer = sample_buffer.detach()
                    xguess = xguess.detach()
                    raw = raw.detach()
                    if render_times > 1:
                        fp = fp.detach()
                        sample_block_buffer = sample_block_buffer.detach()
                    speed = (time.time()-start_time)/((n+1)*render_count)

                    print(f"[TRAIN] Frame:[{frame+1}/{frames}] RawFrame:{raw_frame} Epoch:[{epoch+1}/{epochs}] Iter:[{n+1}/{iters}] Render Bunch:[{i+1}/{render_count}] MSE:{loss_mse.item():.5g} Loss:{loss.item():.5g} Speed:{speed:.3g} s/it")

        ut.save_results(raw, save_dir, projection_name+'_frame%.3d.tif' % (raw_frame))


if __name__ == '__main__':
    cfg = parse_args()
    if cfg.is_dynamic:
        if cfg.mode != 'intensity':
            raise ValueError('is_dynamic=true only supports mode="intensity"')
        run_dynamic(cfg)
    elif cfg.mode == 'intensity':
        run_intensity(cfg)
    elif cfg.mode == 'phase':
        run_phase(cfg)
    else:
        raise ValueError('Unknown mode: %s (expected "intensity" or "phase")' % cfg.mode)
