# AutoOffload.jl

AutoOffload.jl is an experimental library looking into automatic offloading
of costly computations to accelerators like GPUs for user-friendly speedups.
While not as efficient as an algorithm fully designed to stay on an accelerator
due to the data syncing, for costly operations, like matrix multiplications
and FFTs, this can give a sizable speedup. The purpose of this library is
to attempt to automatically determine cutoff points for which offloading
to an accelerator makes sense, and then utilize this so that all other libraries
auto-GPU/TPU/distribute/etc. when appropriate.

## Design Goal

The goal is to have an `autotune()` function which runs some benchmarks to
determine optimal cutoff values for your hardware configuration, and from this
setup internal calls so that acclerated versions will auto-offload. The calls
are all appended with `accelerated`, like:

- `accelerated_mul!`
- `accelerated_fft`
- `accelerated_ldiv!`

This library is made to be automatic, using compile-time checking to enable
offloads based on installed compatible packages, but not require any special
dependencies. This means that a library is safe to depend on and use AutoOffload.jl
for the `accelerated` functions without getting a dependency on the GPU/TPU/etc.
libraries.

## Pirate Mode

We plan to implement a pirated version, so that `using AutoOffload.Pirate`
will replace the common `*`, `mul!`, etc. calls with the accelerated versions,
which will allow auto-acceleration in libraries which have not been setup with
the `accelerated` interface functions.
