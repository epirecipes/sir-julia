# sir-julia
Various implementations of the classical SIR model in Julia

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

- Ordinary differential equation
- Stochastic differential equation
- Function map
- Stochastic Markov model
- Jump process (Gillespie)
- Discrete event simulation
- Agent-based model

Note that the implementations and choice of parameters may be suboptimal, and are intended to illustrate more-or-less the same underlying biological process with different mathematical representations. I've also tried to transform parameterisations in discrete time as closely as possible to their continuous counterparts. Please see the great work by [Linda Allen](http://www.math.ttu.edu/~lallen/) for how these different representations compare.

## Types of output

- Jupyter notebooks
- GitHub markdown
- PDF
- HTML
- Julia script

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
