# Ozaki Scheme
## High-Precision GEMM via Low-Precision Hardware

LIBXSTREAM and LIBXS

---

## The Problem

Scientific computing needs high precision (fp64), but:

- Modern hardware optimized for low precision (int8, etc.)
- GPU tensor cores: 10-100x faster for int8 vs fp64
- Need accuracy without sacrificing performance

**Solution**: Ozaki decomposition

---

## What is Ozaki?

High-precision GEMM using low-precision arithmetic

Two schemes:
- **Scheme 1**: Mantissa slicing (int8, bf16, etc.)
- **Scheme 2**: Chinese Remainder Theorem (CRT)

Intercepts BLAS GEMM calls transparently via `LD_PRELOAD`

---

## Quick Architecture

```
Application calls DGEMM/SGEMM or ZGEMM/CGEMM
            |
            v
    libwrap.so intercepts
            |
            v
    Decompose to int8 slices/residues
            |
            v
    Low-precision dot products
    (AVX-512 VNNI / GPU DPAS)
            |
            v
    Reconstruct high-precision result
```

---

## Getting Started

### Build (5 minutes)

```bash
cd $HOME
git clone https://github.com/hfp/libxstream.git
cd libxstream
make -j $(nproc)

cd $HOME
git clone https://github.com/hfp/libxs.git
cd libxs
make -j $(nproc)

cd samples/ozaki
make -j $(nproc)
```

---

## Build Output

Produces:
- `libwrap.so` - Shared library for `LD_PRELOAD`
- `libwrap.a` - Static library for `--wrap` linking

Test drivers:
- `dgemm-wrap.x`, `sgemm-wrap.x`, and
- `zgemm-wrap.x`, `cgemm-wrap.x`

---

## First Test Run

```bash
cd samples/ozaki

# Run test with 256x256 matrix
./dgemm-wrap.x 256
```

Expected output:
```
Ozaki-Scheme Low-Precision GEMM
GEMM: NN M=256 N=256 K=256
ozaki_oz1: nslices=8 block=16x16x16 vnni=1
Time: 0.123 seconds
```

---

## Deploy with Your Application

### Method 1: LD_PRELOAD (Linux)

```bash
export LD_PRELOAD=/path/to/libwrap.so
./your_application
```

All DGEMM/SGEMM/ZGEMM/CGEMM calls intercepted automatically

**No recompilation needed!**

---

## Critical: MPI + LD_PRELOAD

### Wrong (may not work)

```bash
LD_PRELOAD=/path/to/libwrap.so mpirun -np 4 ./app
```

or

```bash
export LD_PRELOAD=/path/to/libwrap.so
mpirun -np 4 ./app
```

Only `mpirun` gets wrapper, not the actual MPI ranks!

---

## Critical: MPI + LD_PRELOAD

### Correct

```bash
mpirun -np 4 env LD_PRELOAD=/path/to/libwrap.so ./app
```

**The wrapper must be visible to spawned MPI ranks.**

---

## MPI Implementation Specifics

**OpenMPI**:
```bash
mpirun -np 4 -x LD_PRELOAD=/path/to/libwrap.so ./app
```

**Intel MPI**:
```bash
mpiexec -n 4 -genv LD_PRELOAD /path/to/libwrap.so ./app
```

**SLURM srun**:
```bash
srun -n 4 --export=ALL,LD_PRELOAD=/path/to/libwrap.so ./app
```

---

## Complete SLURM Example

```bash
#!/bin/bash
#SBATCH -o %x-%j.txt
#SBATCH -J yourapp
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --ntasks-per-node=16
#SBATCH -N 1

export LD_PRELOAD=$HOME/libxs/samples/ozaki/libwrap.so
export OZAKI_VERBOSE=1  # Stats at end (default: none!)
export OZAKI_RSQ=0.9    # Dump if wrong
export OZAKI=2          # CRT scheme

# Run
srun --export=ALL ./yourapp.x workload.inp
```

---

## Scheme 1: Mantissa Slicing

Decomposes IEEE-754 mantissa into 7-bit int8 slices

```
fp64 (52 bits) -> 8 slices x 7 bits (by default)
fp32 (23 bits) -> 5 slices x 7 bits (by default)
```

Compute S*(S+1)/2 pairwise int8 dot products

Uses AVX-512 VNNI (`VPDPBUSD`) when available

---

## Scheme 2: CRT

Each element reduced modulo small coprimes (<= 128)

```
fp64 -> 19 modular int8 residues (by default)
fp32 -> 10 modular int8 residues (by default)
```

**Linear cost** in number of primes (vs quadratic for slices), but  
relatively expensive reconstruction

Reconstruct via Garner's algorithm + Horner evaluation

---

## Accuracy and Performance

**`OZAKI=1`** (default):
- Moderate accuracy (`OZAKI_N=8` slices)
- More accurate: `OZAKI_FLAGS=0`

**`OZAKI=2`** (CRT):
- High accuracy (`OZAKI_N=19` coprimes)

---

## Key Environment Variables

```bash
# Accuracy vs performance
export OZAKI_TRIM=7         # Omitted "levels"
export OZAKI_CACHE=1        # Speedup transfer

# Debug
export OZAKI_VERBOSE=1      # Print stats at exit

# Performance (GFLOPS/s)
export OZAKI_PROFILE=1      # Device performance
export OZAKI_PROFILE=2      # Main kernel only
```

---

## Validation

Always validate first deployment

```bash
export OZAKI_VERBOSE=1      # Summary at termination
export OZAKI_VERBOSE=100    # Dump every 100th GEMM
```

Exits if threshold exceeded (`OZAKI_EXIT=1` is default)

```bash
export OZAKI_RSQ=0.99       # 1: correct (prefer)
export OZAKI_EPS=0.01       # 0: correct (!= 1-RSQ)
```

Output to stderr

```
GEMM: ncalls=90620954 linf=0 linf_rel=0 l2_rel=0 eps=0.000000 rsq=0.000000
```

---

## Troubleshooting

- Segfault? Application may statically link BLAS
- Use `--wrap` method instead:
  ```bash
  gcc -o app app.o -lwrap \
    -Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ \
    -llapack -lblas
  ```

---

## GPU Offload (LIBXSTREAM)

If built with LIBXSTREAM (OpenCL support):

```bash
export OZAKI_THRESHOLD=12   # Arithmetic Intensity
export OZAKI_CACHE=1        # Heuristic (non-default)
export OZAKI_OCL=0          # Disable GPU
```

---

```

Automatically detects Intel XMX hardware and uses DPAS (Data Processing Accelerator Systolic)

Falls back to CPU if GPU unavailable

---

## Performance Tips

### For CPU workloads:
```bash
export OZAKI=1              # Mantissa slicing
export OZAKI_FLAGS=0        # All-to-all slices
export OZAKI_TRIM=0         # Exact (default)
```

### For GPU workloads:
```bash
export OZAKI=2              # CRT for accuracy
export OZAKI_TRIM=7         # Maybe for Ozaki-1
export OZAKI_OCL=0          # CPU execution
```

---

## Complex GEMM (ZGEMM/CGEMM)

Automatically uses 3M (Karatsuba) method:

```
ZGEMM decomposed into 3 real GEMMs:
- P1 = Re(A) * Re(B)
- P2 = Im(A) * Im(B)
- P3 = (Re+Im)(A) * (Re+Im)(B)

Result:
- Re(C) = P1 - P2
- Im(C) = P3 - P1 - P2
```

Each real GEMM goes through Ozaki wrapper

---

## Debug: Matrix Dumps

Dump problematic matrices for offline analysis:

```bash
export OZAKI_EPS=1e-6       # Dump if error > threshold
export OZAKI_RSQ=0.9        # Dump if R^2 < threshold
```

Creates:
- `gemm_<slurm-rank>-a.mhd`, and
- `gemm_<slurm-rank>-b.mhd`

Reload and test:
```bash
./dgemm-wrap.x gemm-292284-0-a.mhd gemm-292284-0-b.mhd
```

---

## Resources

**Code**:
- LIBXS: https://github.com/hfp/libxs
- LIBXSTREAM: https://github.com/hfp/libxstream (GPU offload)

**Documentation**:
- API Reference: https://libxs.readthedocs.io/

**Support**:
- GitHub Issues: https://github.com/hfp/libxs/issues

---

## Summary

1. Build: `cd libxs/samples/ozaki && make -j`
2. Deploy: `mpirun env LD_PRELOAD=./libwrap.so ./app`
3. Validate: `export OZAKI_VERBOSE=-1`
5. Debug: `export OZAKI_RSQ=0.95`

**Key takeaway**: Correct MPI + LD_PRELOAD order!

---

## Questions?

**Contact**:
- Hans Pabst: hans.pabst @ intel.com
- GitHub: https://github.com/hfp/libxs

Thank you!
