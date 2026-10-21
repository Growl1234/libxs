# Registry Microbenchmark

Benchmarks the performance of the LIBXS registry (key-value store)
dispatch path under different access patterns and concurrency levels.

## Building

```bash
cd samples/registry
make
```

Produces `registry.x` (C). The Fortran variant `registryf.x` is built only
when a Fortran compiler is found and the library was built beforehand: without
a built library the sample is compiled header-only and the Fortran variant is
skipped.

## Programs

### registry.x (C)

```bash
./registry.x [total [nrepeat [nthreads [keysize]]]]
```

| Argument | Default       | Description                               |
|----------|---------------|-------------------------------------------|
| total    | 10000         | Number of unique keys to register (min 2) |
| nrepeat  | 10            | Repeat iterations for lookup phases       |
| nthreads | max available | Number of OpenMP threads                  |
| keysize  | sweep         | Pin one key size in Bytes (8 to 64)       |

Without `keysize`, the single-threaded phases sweep key sizes of 8, 12, 16,
28, 32, 48 and 64 Bytes and print one row per size. A pinned `keysize` prints
the per-phase detail and the registry's size, capacity and footprint instead.
The multi-threaded phases use the pinned size, or 12-Byte keys by default.

Measurements:

- Duration to register (insert) all keys into the registry.
- Cold lookup with shuffled access pattern.
- Locked lookup with sequential/repeating pattern (small working set).
- Multi-threaded parallel reads across all threads.
- Contended parallel writes (each thread registers its own key range).
- Mixed read/write: one writer thread while remaining threads read.

The multi-threaded benchmarks require at least 2 threads and are
skipped when running single-threaded.

### registryf.x (Fortran)

```bash
./registryf.x
```

Fortran variant with hardcoded parameters (10000 keys, 10 repeats).
Measures registration, cold lookup, and locked lookup.
