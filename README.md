# IXI skull stripping on Goethe

Scripts and notes for extracting the brain from IXI T1 MRI volumes on the
Goethe University cluster, using SynthStrip in an Apptainer container.

The run completed on 21 September 2026. It processed 577 T1 volumes from
576 subjects: 576 cropped volumes and one additional raw scan of IXI002.
Every input has a brain image, a binary mask, a processing log, and a JSON
record of the input, container, arguments, and validation results.

The earlier worklog ended at a failed container build. That problem was
resolved before the completed run described here.

## Results and locations

The project directory on Goethe is `/work/dldevel/momeni`. It resolves to
`/panfs/vdura1/dldevel/momeni` on the filesystem used for this work.

| Output | Count | Path relative to the project directory |
| --- | ---: | --- |
| Brain images from cropped inputs | 576 | `datasets/IXI_skullstripped/derivatives/mri_crop/<subject>/<subject>-T1_brain.nii.gz` |
| Masks from cropped inputs | 576 | Same subject folders, ending in `_mask.nii.gz` |
| Brain image and mask from raw IXI002 | 2 files | `datasets/IXI_skullstripped/rawdata/IXI002/anat/` |
| PNG previews of all brain images and masks | 1,154 | `datasets/IXI_skullstripped/png_previews/` |
| Sample mask overlays | 1 montage | `datasets/IXI_skullstripped/qc_sample.png` |
| Six-subject example montage | 1 montage | `skullstrip_examples.png` |

PNG previews preserve the subject-folder structure. Each preview contains
three slices; it does not contain every slice of the volume. The PNG folder
occupied about 136 MiB when it was generated. The NIfTI files are the
volumetric outputs to use for further analysis.

The repository contains scripts and documentation. MRI data, previews, container
images, the FreeSurfer license, and machine credentials stay on the cluster.

## Input data

The initial setup copied the lab's IXI tree into `datasets/IXI` using `rsync`.
The original symlink was kept separately as `datasets/IXI_galella_link`.
That setup is recorded in the supplied handoff; this project did not download
a new native-resolution IXI cohort.

The processing manifest is stored on the cluster at
`containers/jobs/ixi_t1_manifest.json`. It records each input's relative path,
file size, and modification time. Selection includes real T1 NIfTI files and
excludes hidden AppleDouble files such as `._IXI002-T1.nii.gz`, symlink files,
and the T2 and PD modalities.

The manifest contains 576 entries under `derivatives/mri_crop` and one under
`rawdata`, for IXI002. Do not treat these as 577 distinct participants.

## Software and computing

| Component | Version or configuration |
| --- | --- |
| Container | `docker://freesurfer/synthstrip:1.8-gpu` |
| SynthStrip model | Version 1, as reported in the processing logs |
| Apptainer | `1.5.0-1.el9` |
| Container Python | `3.10.18` |
| PyTorch | `2.1.2+cu118` |
| CUDA runtime reported by PyTorch | `11.8` |
| Container NumPy | `1.26.4` |
| NiBabel | `5.3.2` |
| Surfa | `0.6.1` |
| GPU used | NVIDIA A100-SXM4-80GB |
| NVIDIA driver in the run log | `580.105.08` |
| SLURM inference request | `gpu2`, QOS `gpu2`, account `dldevel`, NVIDIA constraint, one GPU, four CPUs, 32 GiB RAM, eight-hour limit |
| Preview environment checked during documentation | Python `3.9.25`, NumPy `1.23.5` |
| First example montage | ImageMagick `convert` and `montage`; installed version checked as `6.9.13-52` |

The container versions above were read from the existing GPU image. The GPU,
CUDA, and driver details also appear in `logs/skullstrip-1774631.out`.
Host tool versions were checked on 22 September and are not a complete
environment snapshot from the original run.

The GPU image's recorded SHA-256 is:

```text
d2e0c4f347402348650ca23e73458d1ccc4120b5d1f8ea34206a9b1c2704f826
```

The image tag is not an immutable version lock. The saved SIF hash identifies
the image actually used; rebuilding from the tag later may produce a different
file. A new hash also prevents the runner from silently reusing results from
the old image.

## Container build

The first build attempts failed when PRoot tried to use `ptrace`. Changing the
temporary directory and setting `PROOT_NO_SECCOMP=1` did not resolve it.
The working build set `APPTAINER_IGNORE_PROOT=1` and limited SquashFS creation
to four processors and 1 GiB of memory:

```bash
bash /work/dldevel/momeni/containers/jobs/build_synthstrip.sh
```

The script builds `containers/synthstrip_1.8-gpu.sif` on the login node.
No GPU allocation is needed for this step. The earlier `1.8` image was
CPU-only; the completed GPU run used `1.8-gpu`.

## Running the pipeline

These scripts retain the paths and scheduler settings used on Goethe.
They are a record of the working deployment, not a portable installer.
A checkout in another directory will not redirect their hardcoded paths.
Before using them elsewhere, update the `WORK` paths in the shell and Python
files, the SLURM account/partition/output directives, and the plotting
environment used by `verify_ixi.sbatch`.

On the existing deployment, the data, manifest, GPU container, scripts, and a
readable FreeSurfer license at `$HOME/licenses/license.txt` must be present.
The license is mounted read-only at `/licenses/license.txt`, with `FS_LICENSE`
set inside the container. No license contents belong in this repository.

For a new input cohort, `ixi_synthstrip.py --prepare` creates the manifest.
It refuses to replace an existing manifest. Keep the current manifest for
the completed cohort: its file sizes and modification times are checked on
every run.

Submit one input first, then the full run:

```bash
bash /work/dldevel/momeni/containers/jobs/submit_ixi.sh --limit 1
bash /work/dldevel/momeni/containers/jobs/submit_ixi.sh
```

The runner calls:

```bash
mri_synthstrip -i INPUT.nii.gz -o BRAIN.nii.gz -m MASK.nii.gz -g -t 4 -f 0
```

It checks CUDA before processing, writes zero outside the mask, and uses the
default brain boundary. Each result is validated before the temporary output
is moved into place. A file lock prevents concurrent runs from writing to the
same output tree. Existing results are reused only when the input record and
container hash match and validation passes. Three consecutive failures stop
the run; each individual call has a 900-second timeout.

No model training, sharpening, or denoising was performed.

## Verification

The full run and a later CPU job checked:

- Matching three-dimensional input, brain, and mask shapes.
- Matching spatial transforms, with absolute affine tolerance `1e-4`.
- Finite voxel values and a binary, nonempty mask that does not fill the volume.
- Preserved intensities inside the mask (`rtol=1e-5`, `atol=1e-4`).
- Exactly zero intensities outside the mask.
- Matching input records and container hashes.
- Exact coverage of the T1 input manifest in the full parallel verification.

| Job | Purpose | Result | Elapsed time |
| --- | --- | --- | --- |
| `1774628` | One-input GPU test | 1/1 passed | 15 seconds |
| `1774631` | Full GPU run | 577/577 passed | 1 hour, 1 minute, 19 seconds |
| `1774636` | Full CPU verification and sample QC export | 577/577 passed | 55 seconds |

All three jobs finished with exit code `0:0`. The full run reused the first
test output and processed the remaining inputs. Separate CPU checkpoint jobs
also checked blocks of 100 results. See [the worklog](docs/WORKLOG.md) for the
checkpoint IDs and a record of the earlier failures.

The CPU verification script requests eight CPUs and 16 GiB RAM. SLURM
accounting reports 80 allocated CPUs for job `1774636`; the Python worker
count follows `SLURM_CPUS_PER_TASK`, not that accounting total.

To verify the current deployment again:

```bash
bash /work/dldevel/momeni/containers/jobs/verify_ixi.sh
# Alternatively, submit parallel verification and the sample overlay export:
sbatch /work/dldevel/momeni/containers/jobs/verify_ixi.sbatch
```

The reports are in `datasets/IXI_skullstripped/run-1774631.json` and
`datasets/IXI_skullstripped/verification-1774636.json`. Summary fields are
included in [results/run_summary.json](results/run_summary.json).

These checks establish file integrity and consistency with the inputs.
They do not establish that every brain boundary is anatomically correct.
The overlay montage is a sample, not a completed visual review of all subjects.

## PNG conversion

The preview converter requires Python 3.9 or later and NumPy. It reads scalar
NIfTI-1 files directly and writes grayscale PNGs using the Python standard
library. NiBabel, Pillow, and Matplotlib are not required by this converter.

From the repository directory:

```bash
python3 tools/nifti_to_png.py \
  /work/dldevel/momeni/datasets/IXI_skullstripped \
  /work/dldevel/momeni/datasets/IXI_skullstripped/png_previews \
  --workers 4
```

Use `--kind brain` or `--kind mask` to select outputs. `--size 512` creates
larger panels. Existing PNGs are skipped unless `--overwrite` is supplied.
The completed conversion produced 1,154 PNGs with no reported failures.

By default each view fits within 256 by 256 pixels; the three-panel PNG is
776 by 256 pixels. The script chooses the middle of the nonzero extent along
each stored voxel axis, applies intensity windowing, and uses nearest-neighbor
resizing. Pixel spacing is used to preserve the panels' physical proportions.
For files with more than three dimensions it uses only the first volume.

There is an orientation limitation: the converter uses array axes directly
and does not interpret the NIfTI affine. Its sagittal/coronal/axial names assume
that the stored axes already have that order. This is not guaranteed for
arbitrary scans, especially native data. Use a NIfTI viewer or the NiBabel-based
`ixi_qc.py` overlay workflow to check orientation. PNGs are quick previews,
not a lossless representation of the full MRI data.

The earlier six-subject montage script, `tools/render_nifti_montage.py`, also
requires ImageMagick. It generated `skullstrip_examples.png` for IXI002,
IXI115, IXI230, IXI342, IXI450, and IXI563. It uses stored-axis slices too.

## Image sharpness

For IXI002, a direct numerical check found that the cropped stripped output
was exactly the cropped input multiplied by its binary mask: the maximum
absolute difference was zero. The full-cohort validation checked preserved
intensities to the tolerances listed above.

The inspected cropped inputs use a 160 by 224 by 160 grid with 1 mm spacing.
Raw IXI002 uses a 256 by 256 by 150 grid with approximately
0.9375 by 0.9375 by 1.2 mm spacing. These are different sampling grids;
the smaller matrix alone does not prove a loss of spatial resolution.

Earlier explanations attributed the apparent blur to preprocessing too
confidently. The checks show that the skull-stripping output preserved the
retained input voxels. They do not identify the exact source of perceived
softness. Earlier resampling, acquisition quality, and viewer interpolation
or contrast settings are possible contributors. The preceding `mri_crop`
processing recipe was not established during this work.

Only IXI002 has a raw T1 in the processed input tree. A native-data comparison
for the other subjects requires obtaining their original scans. No sharpening
filter has been applied to these NIfTIs.

## Files in this repository

- `containers/jobs/`: container build, SLURM submission, processing,
  verification, and QC scripts used on Goethe.
- `tools/nifti_to_png.py`: recursive PNG preview conversion.
- `tools/render_nifti_montage.py`: initial six-subject preview.
- `docs/WORKLOG.md`: setup history and work completed after the handoff.
- `results/run_summary.json`: summary fields extracted from the saved reports.

`watch_ixi_checkpoints.sh` is retained as a historical script. It contains the
completed GPU job ID and checks the 200–500 milestones; the first 100-case
checkpoint was submitted separately. Update its job ID and state directory
before using it for a different run.

## References

- [SynthStrip documentation](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
- [SynthStrip paper](https://doi.org/10.1016/j.neuroimage.2022.119474)
- [IXI dataset](https://brain-development.org/ixi-dataset/)

The software versions, job results, counts, and checks in this README were
assembled from the local scripts, processing logs, JSON reports, and SLURM
accounting records. Documentation last checked: 22 September 2026.
