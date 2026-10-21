# Synchronization Primitives

Two binaries over `libxs_sync.h`: `sync_lock.x` for the lock kinds and
`sync_barrier.x` for `libxs_barrier_t`.

## sync_lock.x

Micro-benchmark for the lock implementations provided by LIBXS
(`libxs_sync.h`). Measures single-thread latency (uncontended
acquire/release) and multi-thread throughput (mixed read/write
workload) for each lock kind:

| Lock kind            | Description                                      |
|----------------------|--------------------------------------------------|
| LIBXS_LOCK_DEFAULT   | Compile-time default (typically atomic)          |
| LIBXS_LOCK_SPINLOCK  | OS-native or CAS-based spin lock (if available)  |
| LIBXS_LOCK_MUTEX     | OS-native mutex / pthread_mutex_t (if available) |
| LIBXS_LOCK_RWLOCK    | Reader/writer lock / pthread_rwlock_t            |

All four kinds are always benchmarked. Platform support only decides
whether a kind is the OS-native primitive or LIBXS's own implementation,
which the output labels "OS-native" or "Other".

## Build

```bash
cd samples/sync
make
```

OpenMP is enabled by default (OMP=1) for multi-threaded tests.

## Run

```bash
./sync_lock.x [nthreads] [wratio%] [work_r] [work_w] [nlat] [ntpt]
```

| Argument | Default       | Description                                      |
|----------|---------------|--------------------------------------------------|
| nthreads | all available | Number of OpenMP threads                         |
| wratio%  | 5             | Percentage of write operations (0-100)           |
| work_r   | 100           | Simulated work inside read-critical section (cy) |
| work_w   | 10 * work_r   | Simulated work inside write-critical section     |
| nlat     | 2000000       | Iterations for latency measurement               |
| ntpt     | 10000         | Iterations per thread for throughput measurement |

### Example

```bash
./sync_lock.x 4 5 100 1000
```

Each measured line names the lock kind it belongs to. The run first prints
the pause budgets it was built with (lock and barrier), which a contended
figure depends on; compare runs only when they match.

For a short, write-only critical section, as taken by a hot internal lock:

```bash
./sync_lock.x 16 100 1 1 100000 5000
```

## Measurement Details

- RO-latency: uncontended read-lock acquire/release pairs (4x
  unrolled), reported as nanoseconds per operation and TSC cycles.
- RW-latency: uncontended write-lock acquire/release pairs (4x
  unrolled).
- Throughput: all threads run a mixed read/write workload governed by
  wratio%. Simulated work inside the critical section is subtracted
  so only synchronization overhead is reported.

## sync_barrier.x

Cost of a rendezvous over `libxs_barrier_t`, and of the broadcast that
hands one task's value to the rest.

```bash
./sync_barrier.x [nthreads] [nrepeat]
```

| Argument | Default       | Description                          |
|----------|---------------|--------------------------------------|
| nthreads | all available | Tasks in the team                    |
| nrepeat  | 100000        | Rendezvous per measurement           |

The output gives the cost per rendezvous in nanoseconds and cycles, once
for `wait:` (plain rendezvous) and once for `bcast:` (rendezvous with a
broadcast value).

The team size is taken from the runtime, which may grant fewer threads than
requested. The stamp each task writes per rendezvous is checked in an
untimed pass and every broadcast value is checked; the binary fails,
reporting stale stamps or wrong broadcasts, if one is wrong. Expect the
barrier to cost more once the team exceeds the number of cores.
