# IXI processing worklog

This record combines the supplied setup handoff with the subsequent cluster
work. Completed-run claims were checked against local files and job records.
The initial license and dataset-copy history comes from the supplied handoff.

## Initial setup

The original setup established the FreeSurfer license path and copied the
shared lab IXI tree into `/work/dldevel/momeni/datasets/IXI` with `rsync`.
The previous symlink was retained as `datasets/IXI_galella_link`.
The handoff described the copied dataset as about 40 GiB; that is a historical
estimate, not a new measurement made for this README.

The initial SynthStrip build failed at SquashFS creation because PRoot's
`ptrace(TRACEME)` call was denied. Moving temporary files and setting
`PROOT_NO_SECCOMP=1` did not fix the problem. An initial GPU job (`1774566`)
waited for the missing image and was cancelled, according to the handoff.
The build attempted on `gpu_test` in job `1774592` failed with the same error;
the saved error log confirms it.

The handoff's statements that the build remained blocked and the output
directory was empty describe that earlier stage. They are no longer the
project's status.

## Work completed after the handoff

### Container and GPU setup

The working build disabled PRoot through `APPTAINER_IGNORE_PROOT=1` and passed
`-processors 4 -mem 1G` to SquashFS. The container was built on the login node.
The CPU-only image was retained locally, and the processing scripts switched
to `freesurfer/synthstrip:1.8-gpu` for CUDA inference.

The submission scripts were changed to check that the image and license exist
before requesting a GPU. They also check the image's CUDA-enabled PyTorch.
The job requests the `gpu2` partition with an NVIDIA constraint and passes GPU
visibility into Apptainer using `--nv`. A CUDA operation is performed before
any input is processed.

### Input manifest and processing

`ixi_synthstrip.py` freezes the input list, records sizes and modification
times, and mirrors the input folders under `IXI_skullstripped`. The cohort is
576 cropped T1s plus the raw IXI002 T1. Hidden metadata files and other MRI
modalities were excluded.

The runner writes a brain NIfTI, a mask NIfTI, a log, and a provenance JSON
file for each input. The JSON records the SIF hash, command arguments, shape,
mask voxel count, mask fraction, and mask volume. Outputs are validated in a
temporary folder before they are moved into their final paths.

The runner supports resuming validated outputs, checks that inputs have not
changed, locks the output directory against concurrent writers, and stops
after three consecutive failures.

### Run and checkpoint checks

| Job | Work | Outcome |
| --- | --- | --- |
| `1774628` | One-input GPU test | 1/1 passed; 15 seconds total job time |
| `1774631` | Full GPU run | 577/577 passed; 01:01:19 total job time |
| `1774637` | First 100 outputs | 100/100 passed |
| `1774641` | Inputs 101–200 | 100/100 passed |
| `1774642` | Inputs 201–300 | 100/100 passed |
| `1774643` | Inputs 301–400 | 100/100 passed |
| `1774650` | Inputs 401–500 | 100/100 passed |
| `1774636` | Full CPU verification and QC export | 577/577 passed; 55 seconds total job time |

The full GPU run used an NVIDIA A100-SXM4-80GB on `gpu30-022`, with four
allocated CPUs and 32 GiB requested RAM. The wall times above come from SLURM
accounting and include job overhead, not only model inference.

`verify_ixi_parallel.py` rereads the inputs, masks, outputs, and provenance
records. It checks that the manifest covers the current T1 inputs exactly.
It reuses the same validation function as the runner, so this is a separate
read/verification pass, not an independent anatomical evaluation algorithm.

### Visual checks and PNGs

`ixi_qc.py` uses NiBabel to reorient sample arrays to the closest canonical
orientation, exports central slices with masks, and uses Matplotlib to draw
mask outlines. Sample selection includes the first, middle, and last records
and the smallest and largest mask volumes, with duplicate selections removed.
The figure is stored in `datasets/IXI_skullstripped/qc_sample.png`.

The initial six-subject preview used NumPy, a small NIfTI reader, and
ImageMagick. The reusable `nifti_to_png.py` converter then produced one
three-panel grayscale PNG per brain or mask NIfTI. It used four CPU workers,
reported 1,154 successful conversions, and wrote about 136 MiB of previews.
A single-file test was inspected before the full conversion; the script was
syntax-checked and output dimensions were checked with ImageMagick.

The preview converter was written without NiBabel or Pillow because they were
not available in the system Python used for those previews. The container
does include NiBabel. The converter's direct reader handles scalar NIfTI-1
data, not every NIfTI variant. It does not resolve anatomical orientation
from the affine; this limitation is now explicit in the README.

### Sharpness investigation

The raw IXI002 header and several cropped/stripped headers were inspected.
The raw and cropped IXI002 volumes have different shapes and voxel spacings.
The cropped stripped IXI002 data matched the masked cropped input exactly in
a direct float32 comparison. This supports preservation of retained voxel
values during stripping.

The earlier claim that resampling definitely caused the perceived blur was
too strong. The earlier preprocessing code and a controlled sharpness
comparison were not available. The README now distinguishes the measured
voxel equality from possible explanations involving preprocessing, acquisition,
and display settings.

## Tools used

- Bash and Python for job wrappers, file inspection, and processing scripts.
- Apptainer for the CPU and GPU SynthStrip images.
- SLURM (`sbatch`, `squeue`, and `sacct`) for submission, monitoring, and
  accounting; `nvidia-smi` for the GPU and driver record.
- SynthStrip and PyTorch/CUDA for inference; NumPy and NiBabel for validation.
- Surfa, installed in the SynthStrip image, as part of its imaging stack.
- Matplotlib in the existing `env/bin/python` environment for mask overlays.
- ImageMagick for the first montage and PNG format/dimension checks.
- Python `gzip`, `struct`, `zlib`, and NumPy for the lightweight PNG converter.
- SHA-256 hashing for the container identity; JSON for manifests and reports.
- Git for preparing repository files. GitHub publishing depends on an
  authenticated connection; preparing these files does not mean they have
  been pushed.

No neural network was trained or fine-tuned. There was no additional denoising,
sharpening, or native-data download during this processing work.

## Checks still needed

Numerical validation does not establish anatomical mask accuracy. A full
visual review of the cortex, skull base, cerebellum, and cropped edges remains
to be done. Native T1s for the other subjects would be needed to compare the
cropped data against their original acquisition grids. Larger PNGs alone
cannot provide that comparison.

## Documentation package

The scripts in this package were copied from the working cluster deployment.
The current README replaces the outdated status and speculative next steps
in the supplied handoff. It includes the completed work, measured results,
software versions, commands, paths, and known limitations.

Raw data, image previews, container images, license contents, credentials,
old failed-job script copies, and unrelated coursework are not part of the
package. The existing cluster files remain in place.
