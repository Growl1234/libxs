# Timer

Header: `libxs_timer.h`

High-resolution timing via a calibrated TSC when available, with an OS real-time clock fallback.

## Types

```C
typedef unsigned long long libxs_timer_tick_t;
```

Integer type representing a single timer sample.

```C
typedef struct libxs_timer_info_t {
  int tsc;   /* non-zero if a calibrated TSC is used */
} libxs_timer_info_t;
```

## Functions

```C
int libxs_timer_info(libxs_timer_info_t* info);
```

Query timer properties. `info->tsc` is non-zero when a calibrated cycle counter is in use: RDTSC (x86), CNTVCT_EL0 (AArch64), or MFTB (PPC64), calibrated at initialization if the counter is invariant (Release builds only). Otherwise `info->tsc` is zero and the OS real-time clock is used. Triggers lazy library initialization if necessary. Returns `EXIT_SUCCESS`, or `EXIT_FAILURE` if `info` is NULL.

```C
libxs_timer_tick_t libxs_timer_tick(void);
```

Sample the current tick of the monotonic timer. The tick unit is platform-specific; convert with `libxs_timer_duration` or `libxs_timer_ncycles`.

```C
libxs_timer_tick_t libxs_timer_ncycles(
  libxs_timer_tick_t tick0, libxs_timer_tick_t tick1);   /* inline */
```

Absolute difference of two ticks (`LIBXS_DELTA`), i.e., `tick1 - tick0` for samples taken in that order. A wrap-around of the counter is not handled.

```C
double libxs_timer_duration(
  libxs_timer_tick_t tick0, libxs_timer_tick_t tick1);
```

Convert a pair of ticks to elapsed wall-clock time in seconds.
