# LF-PINE: Physics-Informed Explicit Modelling for Light-Field Microscopy Reconstruction

LF-PINE is a fully differentiable, self-supervised framework for
high-resolution 3D reconstruction of light-field microscopy (LFM) data — a
computational digital twin of the LFM imaging process. The target scene is
parameterized by an explicit discretized feature volume coupled with a
lightweight decoding MLP, decoupling spatial resolution from network capacity.
A differentiable wave-optical forward model synthesizes multi-view projections
from the reconstructed volume, and the feature volume, the MLP and the Zernike
aberration coefficients are jointly optimized via automatic differentiation.

The framework applies to both single-photon and two-photon light-field
modalities. This repository contains the released code and per-dataset
settings. Measured/simulated light-field data, PSFs and ground-truth volumes
are available separately from the Zenodo record linked below.

A single reconstruction script is provided:

- `recon.py` — unified LF 3D reconstruction, driven by JSON settings files
  (`settings/*.json`). It covers three pipelines, selected by the config keys
  `mode` and `is_dynamic`:
  - `mode="intensity", is_dynamic=false` — **single light-field** 3D
    reconstruction with a hash-table volume and a coarse-to-fine schedule;
  - `mode="phase", is_dynamic=false` — joint intensity + aberrated Zernike
    phase retrieval from complex pupil PSFs;
  - `mode="intensity", is_dynamic=true` — **dynamic (time-series)** LF
    reconstruction. The first frame is reconstructed from scratch; following
    frames warm-start from the previous volume with a partially re-randomized
    hash table and far fewer epochs, using an FFT-cached forward model.

## Environment

- Python >= 3.9, PyTorch >= 2.0 (CUDA recommended; tested with 2.12 + cu126)
- `pip install -r requirements.txt`

## Repository layout

```
recon.py                 # unified LF reconstruction, static + dynamic (JSON-driven)
model.py                 # hash-table volume networks + Zernike phase fitter
loss.py                  # loss functions (mse/fft/mpg-fft/zTV/positivity/NLLP)
Utils/                   # data loading and forward-model utilities
settings/                # per-dataset JSON settings
examples/highna_case8/    # bundled HighNA10 case 8 data
Projections/             # light-field measurements (input data)
  BPAE_cell/             #   real BPAE cell data (35 views, 1024x1024)
  HighNA10/              #   simulated high-NA data, 10 cases (13 views)
  USAF_phase/            #   simulated USAF phase data (RMS 0.1/0.5/1.0)
  sLFM_data_selectR4/    #   simulated sLFM data, 49 selected views
PSF/                     # point spread functions for every dataset
  PSF_zoom8_101dz1_N35/          #   BPAE_cell
  PSF_zoom4_61dz1_N13/           #   HighNA10
  PSFEx_zoom4_31dz1_N13/         #   USAF_phase (complex pupil PSFs)
  simulatedPSF_serenet_selectR4/ #   sLFM_data_selectR4
  Phase/                         #   Zernike basis + GT aberrations for USAF_phase
Samples/                 # ground-truth volumes for the simulated datasets
  highNA_10/             #   GT for HighNA10
  sLFM_GT/               #   GT for sLFM_data_selectR4
  USAF31.tif             #   GT for USAF_phase
Results/                 # reconstruction outputs (created at run time)
```

## Quick start

HighNA10 case 8 is bundled in [`examples/highna_case8/`](examples/highna_case8/),
including the LF measurement, 13 PSFs and ground truth (about 236 MiB).
Run from the repository root:

```bash
python recon.py --config settings/highna_case8.json
```

### Prepare the datasets

Download the datasets from [Zenodo: LF-PINE](https://doi.org/10.5281/zenodo.23034780)
(DOI: `10.5281/zenodo.23034780`). The archive is split into ten files,
`LF-PINE-v0.1.zip.part00` through `LF-PINE-v0.1.zip.part09`; download all
parts and concatenate them in numerical order to reconstruct
`LF-PINE-v0.1.zip` before extracting it.

Copy the extracted `Projections/`, `PSF/` and `Samples/` directories into
this repository root, preserving the subdirectory names shown above.
These contain the measurements, PSFs, Zernike bases and ground-truth
volumes required by the examples. The full dataset collection is distributed
through Zenodo; the bundled case 8 can run with the files already in GitHub.

The `.gitignore` file excludes local datasets, reconstruction outputs and
Python caches from new commits.

### Run reconstruction

Run all commands from this repository root.

```bash
# simulated high-NA focal-plane data
python recon.py --config settings/highna_fp.json

# real BPAE cell data
python recon.py --config settings/bpae_cell.json

# simulated sLFM data
python recon.py --config settings/slfm_selectR4.json

# simulated USAF phase retrieval (RMS1.0; swap the img_path for RMS0.1/0.5)
python recon.py --config settings/phase_usaf31.json

# dynamic time-series reconstruction: enable is_dynamic on any intensity
# config and point img_path to a multi-frame LF stack, e.g.
python recon.py --config settings/bpae_cell.json \
    --set is_dynamic=true --set img_path=your_timeseries.tif \
    --set start_frame=0 --set end_frame=100
```

Results are written to `Results/<date>_LFP_<notes>_<projection>/`: a
checkpoint volume every `epochs//10` epochs, `<name>_recon.tif` at the end,
and — when a ground truth is configured — `best.tif` plus `eval.txt` /
`eval_log.txt` with PSNR/SSIM. The phase mode additionally saves the
retrieved Zernike coefficients (`zernike_coef.txt`) and phase evolution
(`*_phase.tif`).

### Batch / custom runs

Every config key can be overridden on the command line, e.g. reconstruct all
10 HighNA10 cases:

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  python recon.py --config settings/highna_fp.json \
      --set img_path=Projections/HighNA10/${i}_z61.tif \
      --set gt_path=Samples/highNA_10/${i}_z61.tif \
      --set tag=case${i}
done
```

## Settings reference (`settings/*.json`)

| key | meaning |
| --- | --- |
| `mode` | `intensity` (hash volume + coarse-to-fine) or `phase` (+ Zernike retrieval) |
| `is_dynamic` | `true`: reconstruct a frame range with warm restarts (intensity mode only) |
| `img_path` / `psf_dir` / `gt_path` | input LF stack, PSF directory, optional GT volume |
| `psf_zoom` / `img_zoom` | zoom factors of PSF and data; their ratio sets the volume upsampling |
| `upfactor` | extra reconstruction upsampling factor |
| `Nnum` | number of angular views |
| `z_sample` | >0: keep only the central N PSF z-planes |
| `lr`, `epochs`, `coarse_epoch`, `iters` | learning rate, epochs, coarse-to-fine switch epoch, iterations per epoch |
| `mse_weight`, `fft_weight`, `mpg_weight`, `ztv_weight`, `nllp_weight`, `pos_weight` | loss weights (MSE, Fourier-domain L1, mixed Poisson-Gaussian FFT for low-SNR data, z-axis TV, Poisson NLL, positivity) |
| `cutoff_mode` / `cutoff_sigma` | 0/1/2 cutoff blur on the network output; `cutoff_sigma>0` overrides |
| `is_debackground` / `angle_norm` | background subtraction / per-view normalization of the input |
| `sim_mode` | `sum` or `mean` z-integration in the forward model |
| `polynum`, `zernike_*`, `pupilsize`, `psfsize`, `psf_sim_amp`, `phase_z_layers`, `dao_init` | phase-mode settings |
| `start_frame`, `end_frame`, `dynamic_epochs` (default `epochs//10`), `dynamic_fft_weight` (default `4*fft_weight`), `reset_ratio`, `N_sample` | dynamic-mode settings (`is_dynamic=true`) |
| `gpu_id`, `tag`, `seed`, `save_root` | runtime options |
