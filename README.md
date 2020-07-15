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

## Simulation with different types of model

The above process can be represented in different kinds of ways:

- [Ordinary differential equation using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode/ode.md)
- [Ordinary differential equation using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_mtk/ode_mtk.md)
- [Stochastic differential equation using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde/sde.md)
- [Stochastic differential equation using StochasticDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde_stochasticdiffeq/sde_stochasticdiffeq.md)
- [Linear noise approximation (LNA) to the stochastic differential equation](https://github.com/epirecipes/sir-julia/blob/master/markdown/lna/lna.md)
- [Multivariate birth process reparameterisation of the stochastic differential equation](https://github.com/epirecipes/sir-julia/blob/master/markdown/mbp/mbp.md)
- [Function map](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map/function_map.md)
- [Stochastic Markov model](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov/markov.md)
- [Jump process (Gillespie) using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process/jump_process.md)
- [Jump process (Gillespie) using reaction networks from DiffEqBiological.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_diffeqbio/jump_process_diffeqbio.md)
- [Reaction network conversion to ODEs, SDEs and jump process using ModelingToolkit](https://github.com/epirecipes/sir-julia/blob/master/markdown/rn_mtk/rn_mtk.md)
- [Petri net model to ODEs, SDEs, and jump process using Petri.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/pn_petri/pn_petri.md)
- [Petri net model to ODEs, SDEs, and jump process using AlgebraicPetri.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/pn_algebraicpetri/pn_algebraicpetri.md)
- [Jump process (Gillespie) using Gillespie.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_gillespie/jump_process_gillespie.md)
- [Discrete event simulation using SimJulia](https://github.com/epirecipes/sir-julia/blob/master/markdown/des/des.md)
- [Agent-based model using base Julia](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm_vector/abm_vector.md) as well [as using DifferentialEquations](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm_vector_diffeq/abm_vector_diffeq.md)
- [Agent-based model using Agents.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm/abm.md)

## Generating simulated data

We usually do not observe the trajectory of susceptible, infected, and recovered individuals. Rather, we often obtain data in terms of new cases aggregated over a particular timescale (e.g. a day or a week).

- [Simulated data using ODEs](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_simdata/ode_simdata.md)

## Inference

In addition to the above examples of simulation, there are also examples of inference of the parameters of the model using counts of new cases. Although these are toy examples, they provide the building blocks for more complex situations.

- [Point estimates of parameters of the ODE system using Optim.jl and DiffEqParamEstim.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_optim/ode_optim.md)
- [Bayesian estimates of parameters of the ODE system using Approximate Bayesian Computation](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_abc/ode_abc.md)
- [Bayesian estimates of parameters of the ODE system using Turing.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_turing/ode_turing.md)
- [Bayesian estimates of parameters of the ODE system using NestedSamplers.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_nestedsampling/ode_nestedsampling.md)


## Comments on implementations

Note that the implementations and choice of parameters may be suboptimal, and are intended to illustrate more-or-less the same underlying biological process with different mathematical representations. Additional optimizations may be obtained e.g. by using `StaticArrays`.

I've also tried to transform parameterisations in discrete time as closely as possible to their continuous counterparts. Please see the great work by [Linda Allen](http://www.math.ttu.edu/~lallen/) for how these different representations compare.

## Types of output

Thanks to [`Weave.jl`](https://github.com/JunoLab/Weave.jl), Julia Markdown files (in `tutorials/`) are converted into multiple formats.

- [Jupyter notebooks](https://github.com/epirecipes/sir-julia/tree/master/notebook)
- [GitHub Markdown](https://github.com/epirecipes/sir-julia/tree/master/markdown)
- [HTML](https://github.com/epirecipes/sir-julia/tree/master/html)
- [Julia script](https://github.com/epirecipes/sir-julia/tree/master/script)

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

Change to the root directory of the repository and run the following from within Julia.

```julia
include("build.jl")
weave_all()
```

## Acknowledgements

Examples use the following libraries:

- The [`DifferentialEquations.jl`](https://github.com/SciML/DifferentialEquations.jl) ecosystem for many of the examples
- [`SimJulia`](https://github.com/BenLauwens/SimJulia.jl) for discrete event simulations
- [`Agents.jl`](https://github.com/JuliaDynamics/Agents.jl) for agent-based models
- [`Gillespie.jl`](https://github.com/sdwfrost/Gillespie.jl) for the Doob-Gillespie process
- [`Petri.jl`](https://github.com/mehalter/Petri.jl) for the Petri net models
- [`AlgebraicPetri.jl`](https://github.com/AlgebraicJulia/AlgebraicPetri.jl) for a category theory based modeling framework for creating Petri net models
- [`Turing.jl`](https://turing.ml) for inference using probabilistic programs
- [`NestedSamplers.jl`](https://github.com/TuringLang/NestedSamplers.jl) for nested sampling
- [`GpABC`](https://github.com/tanhevg/GpABC.jl) for inference using Approximate Bayesian Computation

Parts of the code were taken from @ChrisRackauckas [`DiffEqTutorials`](https://github.com/SciML/DiffEqTutorials.jl), which comes highly recommended.
