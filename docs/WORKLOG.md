# Worklog

The project followed three main steps:

1. Created a FreeSurfer account and obtained the required license.
2. Built an Apptainer container using the FreeSurfer SynthStrip GPU image.
3. Ran SynthStrip on the IXI T1 MRI volumes to create skull-stripped brain
   images and binary brain masks.

## Setup

Used the existing dataset at `/work/dldevel/galella/datasets/IXI`, copied with
`rsync` into the personal workspace. The original symlink was retained as
`datasets/IXI_galella_link`; the FreeSurfer license stayed outside the repository.

Early builds failed because PRoot's `ptrace` call was denied. The working build
used `APPTAINER_IGNORE_PROOT=1` and SquashFS options `-processors 4 -mem 1G`.
The CPU-only image was replaced for inference by `synthstrip:1.8-gpu`.

## Processing and checks

`ixi_synthstrip.py` records input paths, sizes, and modification times, then
runs `mri_synthstrip -g -t 4 -f 0`. It checks CUDA, validates temporary outputs,
records the container hash, and reuses matching validated results. A lock
prevents concurrent writes; three consecutive failures stop the run.

All 577 inputs passed. Checkpoint jobs 1774637, 1774641, 1774642, 1774643,
and 1774650 each verified 100 outputs. Full CPU verification reread all files
using the same validation function. It requested 8 CPUs and 16 GiB RAM;
SLURM accounting reported 80 allocated CPUs for that job.

## Previews and sharpness

NiBabel and Matplotlib produced sample mask overlays. NumPy and ImageMagick
produced the first montage. The Python/NumPy converter then generated
1,154 PNGs using four workers, occupying about 136 MiB. A sample was visually
inspected and output dimensions were checked.

A numerical comparison of cropped IXI002 showed zero difference between the
stripped image and the masked input. This does not establish the cause of
perceived blur or prove anatomical mask accuracy.

Scripts are in `containers/jobs/` and `tools/`; report summaries are in
[results/run_summary.json](../results/run_summary.json).
The checkpoint watcher retains the old GPU job ID and must be updated before
reuse. No model training, sharpening, or denoising was performed.
