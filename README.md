# IXI skull stripping

Completed on Goethe on 21 September 2026: **577 T1 volumes from 576 subjects**,
producing 577 brain images, 577 masks, and 1,154 PNG previews.

## Steps completed

1. Created a FreeSurfer account and obtained the required license. The license
   is stored privately on the cluster and is not included in this repository.
2. Created an Apptainer container from the FreeSurfer SynthStrip GPU image.
3. Ran FreeSurfer SynthStrip on the IXI T1 MRI data to remove the skull and
   produce a brain image and binary brain mask for each input volume.

## Dataset

The IXI dataset came from `/work/dldevel/galella/datasets/IXI` and was copied
with `rsync` to `/work/dldevel/momeni/datasets/IXI`. This run used the existing
cluster data, with no new download.

Inputs were 576 cropped T1 volumes and one additional raw IXI002 scan.
T2, PD, and hidden metadata files were excluded. The input manifest is stored
on the cluster at `containers/jobs/ixi_t1_manifest.json`.

## Software and computing

| Component | Configuration |
| --- | --- |
| Container | `freesurfer/synthstrip:1.8-gpu`; SynthStrip model version 1 |
| Apptainer | 1.5.0-1.el9 |
| Container Python stack | Python 3.10.18, PyTorch 2.1.2+cu118, CUDA 11.8 |
| Imaging libraries | NumPy 1.26.4, NiBabel 5.3.2, Surfa 0.6.1 |
| GPU | NVIDIA A100-SXM4-80GB; driver 580.105.08 |
| SLURM request | `gpu2` partition/QOS, `dldevel` account, NVIDIA constraint, 1 GPU, 4 CPUs, 32 GiB RAM, 8-hour limit |
| Preview tools | Python/NumPy; Matplotlib for overlays; ImageMagick for the first montage |

Host tools checked on 22 September: Python 3.9.25, NumPy 1.23.5,
ImageMagick 6.9.13-52. The container hash is recorded in
[results/run_summary.json](results/run_summary.json).

## Work completed

- Fixed the PRoot build failure using `APPTAINER_IGNORE_PROOT=1`, then built the GPU container.
- Added input tracking, resumable processing, per-image logs, and output validation.
- Ran a GPU test, the full batch, five checkpoint checks, and full CPU verification.
- Created mask-overlay samples, a six-subject montage, and a reusable PNG converter.

| Job | Purpose | Result | Time |
| --- | --- | --- | --- |
| 1774628 | GPU test | 1/1 passed | 15 seconds |
| 1774631 | Full GPU run | 577/577 passed | 1:01:19 |
| 1774636 | CPU verification and sample QC export | 577/577 passed | 55 seconds |

Validation checked shapes, spatial transforms, finite values, binary masks,
preserved intensities inside the mask, zero background, and input coverage.
A full visual review of anatomical boundaries remains to be done.
See the short [worklog](docs/WORKLOG.md) for implementation notes.

## Output locations

Paths below are relative to `/work/dldevel/momeni`
(also accessible as `/panfs/vdura1/dldevel/momeni`).

| Files | Location |
| --- | --- |
| Cropped brain images and masks | `datasets/IXI_skullstripped/derivatives/mri_crop/<subject>/` |
| Raw IXI002 brain image and mask | `datasets/IXI_skullstripped/rawdata/IXI002/anat/` |
| PNG previews | `datasets/IXI_skullstripped/png_previews/` |
| Sample mask overlays | `datasets/IXI_skullstripped/qc_sample.png` |
| Six-subject montage | `skullstrip_examples.png` |

Brain images end in `_brain.nii.gz`; masks end in `_mask.nii.gz`.
Each has a processing log and provenance JSON alongside it.
Data, previews, containers, and license files stay on the cluster.

## Commands

For the existing Goethe deployment, with the data, manifest, and license at
`$HOME/licenses/license.txt` in place:

```bash
bash /work/dldevel/momeni/containers/jobs/build_synthstrip.sh
bash /work/dldevel/momeni/containers/jobs/submit_ixi.sh --limit 1
bash /work/dldevel/momeni/containers/jobs/submit_ixi.sh
sbatch /work/dldevel/momeni/containers/jobs/verify_ixi.sbatch
```

These scripts use fixed cluster paths and scheduler settings; update them
before running elsewhere. Verification plotting also uses the existing
`/work/dldevel/momeni/env/bin/python` environment.

Convert NIfTIs to PNGs from the repository directory (Python 3.9+ and NumPy):

```bash
python3 tools/nifti_to_png.py \
  /work/dldevel/momeni/datasets/IXI_skullstripped \
  /work/dldevel/momeni/datasets/IXI_skullstripped/png_previews \
  --workers 4
```

Options: `--kind brain`, `--kind mask`, `--size 512`, and `--overwrite`.
Existing PNGs are skipped by default.

## Image quality

Each PNG is a three-slice preview, not an export of every slice. The converter
reads scalar NIfTI-1 files and uses stored voxel axes without checking anatomical
orientation; check orientation in a NIfTI viewer.

For cropped IXI002, stripped voxel values exactly matched the input inside
the mask. The cause of perceived softness was not established; preprocessing
and viewer settings are possible factors. No sharpening or denoising was applied.

## References

[SynthStrip documentation](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) ·
[SynthStrip paper](https://doi.org/10.1016/j.neuroimage.2022.119474) ·
[IXI dataset](https://brain-development.org/ixi-dataset/)
