# Spatial

Samples for spatial data structures and queries built on libxs primitives:
space-filling curves (Hilbert, Morton), kd-tree partition and search, and
spatial pair counting. Each topic uses a filename prefix to group related
files in this flat directory.

| Prefix       | Topic                                    |
|--------------|------------------------------------------|
| `stratify_*` | 3D-to-2D layout via space-filling curves |
| `paircnt_*`  | Spatial pair counting                    |

Underlying libxs primitives (from `libxs_perm.h`):

- Hilbert and Morton curve encode/decode (N-D stratification)
- kd-tree build, partition, nearest-neighbor search
- Sort by spatial locality (various methods)

## Building

The default target builds the C sample `paircnt_dd.x` (header-only when the
library is not built). The Python scripts load the shared library, so build
libxs with `STATIC=0` first so that `lib/libxs.so` exists:

```bash
cd ../..
make PEDANTIC=2 STATIC=0
cd samples/spatial
make
```

The scripts look for the library given by `--libxs PATH`, then by the
environment variable `LIBXS_LIBRARY`, then in `lib/` of the repository, then
on the system search path.

Optional Python adapters need `numpy` and `h5py`; `paircnt_validate.py` also
needs `corrfunc` (which needs `libgsl-dev` to build). All three are listed in
`requirements.txt`:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Stratification (stratify\_\*)

Folding higher-dimensional data into lower-dimensional layouts using
space-filling-curve stratification. The mapping can be computed once for a
fixed detector or simulation grid and reused by any downstream pipeline.

The transform encodes each source voxel coordinate as a 3D Hilbert or Morton
key and decodes it as a 2D coordinate:

```text
3D coordinate -> 3D curve rank -> inverse 2D curve coordinate
```

This preserves the deterministic curve order while giving downstream code
a 2D tensor layout that can be consumed by optimized 2D convolution kernels.

The samples distinguish the curve order from the 2D frame. The default
`compact` frame streams voxels in curve-rank order into a dense near-square
sheet, avoiding unused cells when an exact factor pair is available. The
`canonical` frame decodes the finite 3D curve rank through the corresponding
2D curve, preserving the canonical destination curve coordinate at the cost of
possible empty cells.

### Usage

The `dense3d` target runs the synthetic dense 3D sample:

```bash
make dense3d
make dense3d FRAME=canonical
```

The older `make volume` spelling remains as a compatibility alias.

Direct invocation:

```bash
python3 stratify_dense3d.py --shape 8 16 16 --curve hilbert \
  --frame compact --out sheet.pgm
python3 stratify_dense3d.py --shape 8 16 16 --curve morton \
  --frame canonical --map-csv map.csv
```

HDF5 input is optional and requires `h5py` and `numpy`. The input dataset may
be a single `D,H,W` volume, a batched `N,D,H,W` dataset, a channelled
`N,C,D,H,W` or `N,D,H,W,C` dataset, or a flattened `N,V` dataset when
`--hdf5-reshape D H W` is supplied. The default `auto` layout recognizes the
3DGAN Caffe sample layout `N,C,D,H,W`:

```bash
python3 stratify_dense3d.py --hdf5 /tmp/3Dgan-risk-audit/caffe/train.h5 \
  --hdf5-dataset ECAL --hdf5-event 0 --hdf5-channel 0 \
  --curve hilbert --frame compact --out sheet.pgm --out-hdf5 sheet.h5
```

The Makefile exposes the same path without making the default target depend on
external data:

```bash
make hdf5 HDF5_FILE=/tmp/3Dgan-risk-audit/caffe/train.h5 FRAME=compact
```

For flattened HDF5 files, pass the dataset name, layout, and target 3D shape:

```bash
make hdf5 HDF5_FILE=dataset_2_1.hdf5 HDF5_DATASET=showers \
  HDF5_LAYOUT=flat HDF5_RESHAPE="45 16 9"
```

Public calorimeter HDF5 datasets with a documented download path are available
from the CaloChallenge project. The challenge HDF5 files contain
`incident_energies` and flattened `showers` datasets, so use `--hdf5-dataset
showers` and `--hdf5-reshape D H W` with the geometry stated for the selected
dataset. Useful starting points are:

| Source              | Format / notes                       |
|---------------------|--------------------------------------|
| [CaloChallenge][cc] | Overview, geometry, evaluation.      |
| [Dataset 1][cc1]    | ATLAS photons/pions, flat showers.   |
| [Dataset 2][cc2]    | Electrons, reshape to `45 16 9`.     |
| [Dataset 3][cc3]    | Hi-gran electrons, `45 50 18`.       |
| [Legacy sets][ccl]  | Submitted outputs for repro figures. |

[cc]:  https://calochallenge.github.io/homepage/
[cc1]: https://doi.org/10.5281/zenodo.8099322
[cc2]: https://doi.org/10.5281/zenodo.6366270
[cc3]: https://doi.org/10.5281/zenodo.6366323
[ccl]: https://doi.org/10.5281/zenodo.15961728

Arguments:

| Option           | Description                                 |
|------------------|---------------------------------------------|
| `--shape D H W`  | Volume shape (default `8 16 16`).           |
| `--curve`        | `hilbert` or `morton`.                      |
| `--frame`        | `compact` or `canonical`.                   |
| `--libxs`        | Path to libxs shared library.               |
| `--hdf5`         | Read volume from an HDF5 file.              |
| `--hdf5-dataset` | HDF5 dataset name (default `ECAL`).         |
| `--hdf5-layout`  | `auto`/`dhw`/`ndhw`/`ncdhw`/`ndhwc`/`flat`. |
| `--hdf5-reshape` | Reshape flat data to `D H W`.               |
| `--hdf5-event`   | Event index (default `0`).                  |
| `--hdf5-channel` | Channel index (default `0`).                |
| `--out`          | Write 8-bit PGM of the sheet.               |
| `--map-csv`      | Write `src,z,y,x,dst,v,u` rows.             |
| `--out-hdf5`     | Write sheet+map to HDF5 file.               |

The Makefile targets take these variables:

| Variable           | Default        | Used by                            |
|--------------------|----------------|------------------------------------|
| `PYTHON`           | `python3`      | all script targets                 |
| `FRAME`            | `compact`      | all script targets                 |
| `HDF5_FILE`        | (required)     | `hdf5`                             |
| `HDF5_DATASET`     | `ECAL`         | `hdf5`                             |
| `HDF5_LAYOUT`      | `auto`         | `hdf5`                             |
| `HDF5_RESHAPE`     | (none)         | `hdf5`                             |
| `HDF5_EVENT`       | `0`            | `hdf5`                             |
| `HDF5_CHANNEL`     | `0`            | `hdf5`                             |
| `MEDMNIST3D_NPZ`   | (none)         | `medmnist3d`, `medmnist3d-metrics` |
| `MEDMNIST3D_ROOT`  | `~/.medmnist`  | `medmnist3d`, `medmnist3d-metrics` |
| `MEDMNIST3D_FLAG`  | `organmnist3d` | `medmnist3d`, `medmnist3d-metrics` |
| `MEDMNIST3D_SIZE`  | `28`           | `medmnist3d`, `medmnist3d-metrics` |
| `MEDMNIST3D_SPLIT` | `train`        | `medmnist3d`, `medmnist3d-metrics` |
| `MEDMNIST3D_INDEX` | `0`            | `medmnist3d`, `medmnist3d-metrics` |

### MedMNIST3D

The `stratify_medmnist3d.py` sample reads one standardized MedMNIST3D volume
from an `.npz` file and applies the same 3D-to-2D stratification primitive. It
is a dataset adapter rather than a training script, which keeps the dependency
surface small: only `numpy` is required to parse the MedMNIST file. The sample
does not require PyTorch or the `medmnist` Python package unless you use those
tools separately to download the data.

The official MedMNIST distribution is available from the project page and
Zenodo: [MedMNIST](https://medmnist.com/) and
[Zenodo DOI](https://doi.org/10.5281/zenodo.10519652). The 3D subsets are
`organmnist3d`, `nodulemnist3d`, `adrenalmnist3d`, `fracturemnist3d`,
`vesselmnist3d`, and `synapsemnist3d`.

Direct invocation with an explicit NPZ file:

```bash
python3 stratify_medmnist3d.py --npz ~/.medmnist/organmnist3d.npz \
  --split train --index 0 --curve hilbert --frame compact \
  --out stratified_medmnist3d.pgm \
  --map-csv stratified_medmnist3d.csv \
  --label-csv stratified_medmnist3d_label.csv
```

The Makefile exposes the same path without making the default target depend on
external data:

```bash
make medmnist3d MEDMNIST3D_NPZ=~/.medmnist/organmnist3d.npz FRAME=compact
```

The same NPZ input can be used with the metrics script to report invariants,
locality distortion, and LIBXS fingerprint distances for a selected
volume:

```bash
make medmnist3d-metrics MEDMNIST3D_NPZ=~/.medmnist/organmnist3d.npz \
  MEDMNIST3D_SPLIT=test MEDMNIST3D_INDEX=0 FRAME=canonical
```

If `MEDMNIST3D_NPZ` is omitted, the script looks for a dataset under
`MEDMNIST3D_ROOT` using `MEDMNIST3D_FLAG` and `MEDMNIST3D_SIZE`:

```bash
make medmnist3d MEDMNIST3D_ROOT=~/.medmnist MEDMNIST3D_FLAG=nodulemnist3d
```

Arguments of `stratify_medmnist3d.py`:

| Option        | Description                                        |
|---------------|----------------------------------------------------|
| `--npz`       | MedMNIST3D NPZ file (otherwise `--root`/`--flag`). |
| `--root`      | Directory of NPZ files (default `~/.medmnist`).    |
| `--flag`      | 3D subset (default `organmnist3d`).                |
| `--size`      | Image size used with `--root` (default `28`).      |
| `--split`     | `train`, `val`, or `test` (default `train`).       |
| `--index`     | Sample index within the split (default `0`).       |
| `--curve`     | `hilbert` or `morton`.                             |
| `--frame`     | `compact` or `canonical`.                          |
| `--libxs`     | Path to libxs shared library.                      |
| `--out`       | Write 8-bit PGM of the sheet.                      |
| `--map-csv`   | Write the source-to-destination map as CSV.        |
| `--label-csv` | Write the selected label as CSV.                   |
| `--out-hdf5`  | Write sheet+map to HDF5 file.                      |

### Metrics

The `stratify_dense3d_metrics.py` script reports invariants and locality
distortion for the same synthetic and HDF5 inputs:

```bash
make metrics
python3 stratify_dense3d_metrics.py --hdf5 /tmp/3Dgan-risk-audit/caffe/train.h5 \
  --hdf5-dataset ECAL --curve hilbert
python3 stratify_dense3d_metrics.py --hdf5 dataset_2_1.hdf5 --hdf5-dataset showers \
  --hdf5-layout flat --hdf5-reshape 45 16 9 --curve morton
```

The invariant metrics check that stratification is a lossless layout transform:
the total energy, reconstructed voxel values, and per-axis energy profiles match
after applying the inverse map. The distortion metrics measure what changes for
a convolutional model: how far source 3D grid neighbors move apart in the 2D
sheet, and how often adjacent sheet cells correspond to adjacent source voxels.

The script also reports LIBXS fingerprint metrics. These are compact
multi-order L2 descriptors of value and finite-difference structure. The
`fprint.source_reconstructed.diff` value should be zero for a correct lossless
round trip, while `fprint.source_sheet.diff` summarizes how different the
stratified 2D sheet looks as a structured field.

The metrics script takes `--shape`, `--curve`, `--frame`, `--libxs`, and the
`--hdf5*` options of `stratify_dense3d.py`. A MedMNIST3D volume is selected
with `--medmnist3d-npz FILE`, or with `--medmnist3d` and `--medmnist3d-root`,
`--medmnist3d-flag`, `--medmnist3d-size`; `--medmnist3d-split` and
`--medmnist3d-index` pick the sample. `--csv FILE` writes the metrics as CSV.

### Geometry

The current sample treats the source as a regular `D,H,W` index grid. This is
enough for dense 3DGAN-style arrays and for CaloChallenge datasets after their
flattened `showers` arrays are reshaped with the documented dimensions.

Supporting other detector geometries should be data-driven rather than encoded
as a list of known experiments. A general geometry adapter needs one of the
following descriptions:

| Geometry input             | Role                              |
|----------------------------|-----------------------------------|
| Integer logical coords     | Direct Hilbert/Morton input.      |
| Floating physical coords   | Quantized before stratification.  |
| Adjacency edges (optional) | Neighbor-distortion metrics.      |
| Voxel weights (optional)   | Physics metrics for unequal bins. |

---

## Pair Counting (paircnt\_\*)

Spatial pair counting inspired by Corrfunc: given a point catalog, bin all
pairs by separation distance to compute correlation functions. The approach
uses libxs kd-tree partition for cell decomposition and cell-pair pruning
(minimum-separation skip), with vectorized inner loops for the actual distance
computation and bin assignment.

### Usage

```bash
python3 paircnt_gendata.py --npoints 10000 --boxsize 420.0 --out catalog.dat
./paircnt_dd.x catalog.dat --rmin 0.1 --rmax 25 --nbins 20
python3 paircnt_validate.py catalog.dat
```

`paircnt_dd.x <catalog> [--rmin R] [--rmax R] [--nbins N] [--leaf N]`
reads a catalog of `x y z` lines, or a Fortran fastfood file when the name
ends in `.ff`, and prints one `rmin rmax npairs` row per logarithmic bin.
Timings and the total pair count go to stderr.

| Option      | Default | Description                                  |
|-------------|---------|----------------------------------------------|
| `--rmin`    | `0.1`   | Smallest bin edge (greater than zero).       |
| `--rmax`    | `25.0`  | Largest bin edge (greater than `--rmin`).    |
| `--nbins`   | `20`    | Number of bins (at most 128).                |
| `--boxsize` | `0`     | Periodic box: not supported (rejected).      |
| `--leaf`    | `64`    | Smallest kd-tree leaf (1 to 256).            |

`paircnt_gendata.py` writes uniform points (`--clustered` with `--nclusters`
for clustered ones) with `--npoints`, `--boxsize`, `--seed` (default `42`),
and `--out` (`.ff` selects the fastfood format). `paircnt_validate.py` runs
`paircnt_dd.x` (`--exe`, default `./paircnt_dd.x`) and Corrfunc's
non-periodic `DD` with the same `--rmin`, `--rmax`, and `--nbins` on a text
catalog, compares the counts bin by bin, and exits non-zero on a mismatch.

Design goals:

- Leverage `libxs_kdtree_build` / `libxs_kdtree_partition` for spatial
  decomposition rather than a separate grid structure.
- Cell-pair pruning: skip cell pairs whose minimum separation exceeds the
  largest bin edge.
- Vectorized pair-counting kernel for the inner loop (SIMD where available).
- Periodic and non-periodic boundary conditions.
- Output: binned pair counts DD(r), optionally normalized to xi(r) via
  Landy-Szalay with a random catalog.
