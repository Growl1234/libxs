# Registry Microbenchmark

Benchmarks the performance of the LIBXS registry (key-value store)
dispatch path under different access patterns and concurrency levels.

## Programs

### registry.x (C)

```bash
./registry.x [total [nrepeat [nthreads]]]
```

| Argument | Default       | Description                               |
|----------|---------------|-------------------------------------------|
| total    | 10000         | Number of unique keys to register (min 2) |
| nrepeat  | 10            | Repeat iterations for lookup phases       |
| nthreads | max available | Number of OpenMP threads                  |

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

## Scaling Behavior

All phases pass the registry's lock, which bypasses the thread-local
cache. Reads and writes are hence serialized, and their per-op duration
grows with the number of threads contending for the lock.
