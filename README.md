# sir-julia
Various implementations of the classical SIR model in Julia

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epirecipes/sir-julia/master)

## The model

$$
\frac{dS}{dt} = -\beta c \frac{I}{N} S\
\frac{dI}{dt} = \beta c \frac{I}{N} S - \gamma I \
\frac{dR}{dt} = \gamma I \
N = S+I+R
$$

## Types of model

## Types of output

## Running notebooks

```julia
cd(@__DIR__)
import IJulia
IJulia.notebook(;dir="notebook")
```

## Adding new examples

To add an example, make a new subdirectory in the `tutorials` directory, and add a Julia Markdown (`.jmd`) document to it.

Change to the root directory of the repository and run `julia build.jl` from the command line or `include("build.jl")` from within Julia.
