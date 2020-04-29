# sir-julia
Various implementations of the classical SIR model in Julia

Try the notebooks out in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epirecipes/sir-julia/master?filepath=notebook)

## Model considered

GitHub Markdown doesn't parse equations, so here's a description of the underlying SIR model.

- The ordinary differential equation model considers:
  - Susceptible, S, with initial condition S(0)=990
  - Infected, I, with initial condition, I(0)=10
  - Recovered, R, with initial condition R(0)=10
  - Total population, N=S+I+R=1000
- Susceptible individuals make contacts with others at rate c (=10.0), with the probability of a contact with an infectious person being I/N.  With probability β (=0.05), an infected person will infect a susceptible given a contact.
- Infected individuals recover at a per-capita rate γ (=0.25).

## Types of model

The above process can be represented in different kinds of ways:

- [Ordinary differential equation using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode/ode.md)
- [Stochastic differential equation using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde/sde.md)
- [Stochastic differential equation using StochasticDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde_stochasticdiffeq/sde_stochasticdiffeq.md)
- [Function map](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map/function_map.md)
- [Stochastic Markov model](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov/markov.md)
- [Jump process (Gillespie) using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process/jump_process.md)
- [Jump process (Gillespie) using Gillespie.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_gillespie/jump_process_gillespie.md)
- [Discrete event simulation using SimJulia](https://github.com/epirecipes/sir-julia/blob/master/markdown/des/des.md)
- [Agent-based model using Agents.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm/abm.md)

Note that the implementations and choice of parameters may be suboptimal, and are intended to illustrate more-or-less the same underlying biological process with different mathematical representations. I've also tried to transform parameterisations in discrete time as closely as possible to their continuous counterparts. Please see the great work by [Linda Allen](http://www.math.ttu.edu/~lallen/) for how these different representations compare.

## Types of output

Thanks to [`Weave.jl`](https://github.com/JunoLab/Weave.jl), Julia Markdown files (in `tutorials/`) are converted into multiple formats.

- [Jupyter notebooks](https://github.com/epirecipes/sir-julia/tree/master/notebook)
- [GitHub Markdown](https://github.com/epirecipes/sir-julia/tree/master/markdown)
- [PDF](https://github.com/epirecipes/sir-julia/tree/master/notebook)
- [HTML](https://github.com/epirecipes/sir-julia/tree/master/notebook)
- [Julia script](https://github.com/epirecipes/sir-julia/tree/master/notebook)

## Running notebooks

```sh
git clone https://github.com/epirecipes/sir-julia
cd sir-julia
```

Then launch `julia` and run the following.

```julia
cd(@__DIR__)
import IJulia
IJulia.notebook(;dir="notebook")
```

## Adding new examples

To add an example, make a new subdirectory in the `tutorials` directory, and add a Julia Markdown (`.jmd`) document to it. Set the beginning to something like the following:

```md
# Agent-based model using Agents.jl
Simon Frost (@sdwfrost), 2020-04-27
```

Suggested sections:

- Introduction
- Libraries
- Utility functions
- Transitions
- Time domain
- Initial conditions
- Parameter values
- Random number seed
- Running the model
- Post-processing
- Plotting
- Benchmarking

In addition, an appendix that displays the machine on which the code is run, and the package details,{} can be added using the following code:

```julia
include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()
```

Change to the root directory of the repository and run `julia build.jl` from the command line or `include("build.jl")` from within Julia.

## Acknowledgements

Examples use the following libraries:

- The [`DifferentialEquations.jl`](https://github.com/SciML/DifferentialEquations.jl) ecosystem for many of the examples
- [`SimJulia`](https://github.com/BenLauwens/SimJulia.jl) for discrete event simulations
- [`Agents.jl`](https://github.com/JuliaDynamics/Agents.jl) for agent-based models
- [`Gillespie.jl`](https://github.com/sdwfrost/Gillespie.jl) for the Doob-Gillespie process

Parts of the code were taken from @ChrisRackauckas [`DiffEqTutorials`](https://github.com/SciML/DiffEqTutorials.jl), which comes highly recommended.
