project: LIBXS
author: Hans Pabst
license: <a href="https://libxs.readthedocs.io/LICENSE/">BSD 3-Clause License</a>
summary: Portable C library providing building blocks for memory operations, numerics, synchronization, and more.
project_github: https://github.com/hfp/libxs/
project_download: https://github.com/hfp/libxs/releases/latest/
favicon: ../.theme/img/favicon.png
css: ../.theme/ford.css
output_dir: fortran
src_dir: ../libxs
search: true
macro: __LIBXS

<a href="../fortran/module/libxs.html" target="_blank">Reference Documentation</a>

## Usage

Module LIBXS is part of the library. `make` builds it whenever a Fortran
compiler is found (`FORTRAN=0` disables it), and so does CMake
(`-DLIBXS_FORTRAN=OFF` disables it). After installation, `libxs.mod` is
in `include/libxs`, next to the headers.

```fortran
      USE :: LIBXS, ONLY: libxs_memcmp
```

Module LIBXS_JIT adds GEMM, SYRK, and SYR2K dispatch with the kernel
providers (BLAS, MKL JIT, LIBXSMM) filled in. It is not part of the
library: the application compiles `libxs_jit.F` with its own flags and
uses `USE LIBXS_JIT` in place of `USE LIBXS`. Without `-D__LIBXS`, it
compiles to a BLAS-only module that needs no LIBXS library. See
<a href="../libxs_gemm/#kernel-providers">Kernel Providers</a>.

```bash
$FC -D__LIBXS [-D__MKL] [-D__LIBXSMM] $(pkg-config --cflags libxs) \
  -c $PREFIX/include/libxs/libxs_jit.F
$FC app.o libxs_jit.o $(pkg-config --libs --static libxs) -lblas
```
