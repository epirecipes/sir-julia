# sir-julia
Various implementations of the classical SIR model in Julia

Try the notebooks out in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epirecipes/sir-julia/master?filepath=notebook)

## Model considered

The model equations are as follows.

$$
\begin{align*}
\dfrac{\mathrm dS}{\mathrm dt} &= -\frac{\beta c S I}{N}, \\
\dfrac{\mathrm dI}{\mathrm dt} &= \frac{\beta c S I}{N} - \gamma I,\\ 
\dfrac{\mathrm dR}{\mathrm dt} &= \gamma I, \\
S(t) + I(t) + R(t) &= N
\end{align*}
$$

Here's a description of the underlying SIR model.

- The ordinary differential equation model considers:
  - Susceptible, S, with initial condition S(0)=990
  - Infected, I, with initial condition, I(0)=10
  - Recovered, R, with initial condition R(0)=10
  - Total population, N=S+I+R=1000
- Susceptible individuals make contacts with others at rate c (=10.0), with the probability of a contact with an infectious person being I/N.  With probability β (=0.05), an infected person will infect a susceptible given a contact.
- Infected individuals recover at a per-capita rate γ (=0.25).

There are two types of parameterization commonly used in this project; the 'standard' version, that considers the number of individuals in the S, I, and R groups, and an alternative version, in which the dynamics of transmission (βSI/N) and recovery (γI) are modelled directly, with S, I, and R being calculated based on these dynamics and the initial conditions for S, I and R.

## Simulation with different types of model

The above process can be represented in different kinds of ways:

### Ordinary differential equations

- [Ordinary differential equation using the Euler method](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_euler/ode_euler.md)
- [Ordinary differential equation using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode/ode.md)
- [Ordinary differential equation using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_mtk/ode_mtk.md)
- [Ordinary differential equation using Modia.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_modia/ode_modia.md)
- [Ordinary differential equation using ApproxFun.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_approxfun/ode_approxfun.md)
- [Ordinary differential equation with composition using AlgebraicDynamics.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_algebraicdynamics/ode_algebraicdynamics.md)

### Integral equations

- [Volterra integral equation using the Adomian decomposition method](https://github.com/epirecipes/sir-julia/blob/master/markdown/adomian/adomian.md)

### Stochastic differential equations

- [Stochastic differential equation using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde/sde.md)
- [Stochastic differential equation using StochasticDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde_stochasticdiffeq/sde_stochasticdiffeq.md)
- [Stochastic differential equation using Bridge.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde_bridge/sde_bridge.md)
- [Linear noise approximation (LNA) to the stochastic differential equation](https://github.com/epirecipes/sir-julia/blob/master/markdown/lna/lna.md)
- [Multivariate birth process reparameterisation of the stochastic differential equation](https://github.com/epirecipes/sir-julia/blob/master/markdown/mbp/mbp.md)
- [ODEs of means, variances, etc. through moment closure](https://github.com/epirecipes/sir-julia/blob/master/markdown/momentclosure/momentclosure.md)

### Function maps

- [Function map](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map/function_map.md)
- [Function map using DynamicalSystems.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_dynamicalsystems/function_map_dynamicalsystems.md)
- [Function map using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_mtk/function_map_mtk.md)

### Stochastic Markov models

- [Stochastic Markov model](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov/markov.md)
- [Stochastic Markov model using Soss.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov_soss/markov_soss.md)

### Jump processes

- [Jump process (Gillespie) using DifferentialEquations.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process/jump_process.md)
    - [Jump process with a large number of states - in this case, the number of people some infects - captured by an InfiniteArray](https://github.com/epirecipes/sir-julia/blob/master/markdown/infinite_arrays/infinite_arrays.md)
- [Jump process (Gillespie) using reaction networks from Catalyst.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_catalyst/jump_process_catalyst.md)
- [Jump process (Gillespie) using Gillespie.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_gillespie/jump_process_gillespie.md)
- [Jump process using the Sellke construction](https://github.com/epirecipes/sir-julia/blob/master/markdown/sellke/sellke.md)
- [Jump process using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_mtk/jump_process_mtk.md)
- [Jump process using Fleck.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_fleck/jump_process_fleck.md)

### Finite state projection

- [Solution of the master equation using FiniteStateProjection.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_fsp/ode_fsp.md)

### Petri nets

- [Petri net model to ODEs, SDEs, and jump process using Petri.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/pn_petri/pn_petri.md)
- [Petri net model to ODEs, SDEs, and jump process using AlgebraicPetri.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/pn_algebraicpetri/pn_algebraicpetri.md)

### Stock and flow models

- [Simple stock and flow model to an ODE using StockFlow.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_stockflow/ode_stockflow.md)

### Discrete event simulations

- [Discrete event simulation using SimJulia](https://github.com/epirecipes/sir-julia/blob/master/markdown/des/des.md)

### Agent-based models

- [Agent-based model using base Julia](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm_vector/abm_vector.md) as well [as using DifferentialEquations](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm_vector_diffeq/abm_vector_diffeq.md)
- [Agent-based model using Agents.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/abm/abm.md)
- [Transmission network individual-based model using Pathogen.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sim_pathogen/sim_pathogen.md)

### Other representations

- [Reaction network conversion to ODEs, SDEs and jump process using ModelingToolkit](https://github.com/epirecipes/sir-julia/blob/master/markdown/rn_mtk/rn_mtk.md)

## Optimal control

- [Optimal control of an SIR epidemic with non-pharmaceutical interventions using ODEs and Optimization.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_optimization/ode_lockdown_optimization.md)
- [Optimal control of an SIR epidemic with non-pharmaceutical interventions using ODEs and InfiniteOpt.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_infiniteopt/ode_lockdown_infiniteopt.md)
- [Optimal control of an SIR epidemic with non-pharmaceutical interventions using a function map and JuMP.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_lockdown_jump/function_map_lockdown_jump.md)
- [Flattening the curve of an SIR epidemic with non-pharmaceutical interventions using a function map and JuMP.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_ftc_jump/function_map_ftc_jump.md)
- [Optimal control of an SIR epidemic with vaccination using a function map and JuMP.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_vaccine_jump/function_map_vaccine_jump.md)
- [Optimal control of an SIR epidemic with non-pharmaceutical interventions using a function map and stochastic dual dynamic programming with SDDP.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sddp/sddp.md)

## Composing models

Building models from smaller, re-usable components make it easier to build complex models quickly, and also makes it easier to document the development of these models.

- [Composition of ODE models using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_compose/ode_compose.md)
- [Composition of ODE models using AlgebraicDynamics.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_algebraicdynamics/ode_algebraicdynamics.md)
- [Composition of petri net models using AlgebraicPetri.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/pn_algebraicpetri/pn_algebraicpetri.md)

## Generating simulated data

We usually do not observe the trajectory of susceptible, infected, and recovered individuals. Rather, we often obtain data in terms of new cases aggregated over a particular timescale (e.g. a day or a week).

- [Simulated data using ODEs](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_simdata/ode_simdata.md)

## Use of callbacks

- [Changing parameter values at fixed times e.g. lockdown in an SIR model](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown/ode_lockdown.md)
- [Stopping simulations when infected individuals reach zero in stochastic differential equations](https://github.com/epirecipes/sir-julia/blob/master/markdown/sde_stochasticdiffeq/sde_stochasticdiffeq.md)
- [Scheduling recovery times to model a fixed infectious period](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_delay/jump_process_delay.md)
- [Preventing out-of-domain errors in a sinusoidally forced ODE model](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_bifurcation_bruteforce/ode_bifurcation_bruteforce.md)

## Inference

In addition to the above examples of simulation, there are also examples of inference of the parameters of the model using counts of new cases. Although these are toy examples, they provide the building blocks for more complex situations.

### Deterministic models

- [Point estimates of parameters of the ODE system using Optim.jl and DiffEqParamEstim.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_optim/ode_optim.md)
- [Bayesian estimates of parameters of the ODE system using Approximate Bayesian Computation](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_abc/ode_abc.md)
- [Bayesian estimates of parameters of the ODE system using Turing.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_turing/ode_turing.md)
- [Bayesian estimates of time-varying parameters of an ODE system using Turing.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_inference/ode_lockdown_inference.md)
- [Bayesian estimates of parameters of the ODE system using NestedSamplers.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_nestedsampler/ode_nestedsampler.md)
- [Bayesian estimates of parameters of the ODE system using importance sampling, Markov Chain Monte Carlo and Sequential Monte Carlo with Gen.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_gen/ode_gen.md)
- [Bayesian estimates of parameters of the ODE system using message passing using RxInfer.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_rxinfer/ode_rxinfer.md)

### Stochastic models

- [Bayesian inference of transmission network individual-based model parameters using Pathogen.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/infer_pathogen/infer_pathogen.md)
- [Estimating the likelihood of a discrete-time Markov model using a simple particle filter](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov_pfilter/markov_pfilter.md)
- [Point estimates of parameters of a discrete-time Markov model using Ensemble Kalman Inversion](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov_eki/markov_eki.md)
- [Approximate posterior estimates of parameters of a discrete-time Markov model using Ensemble Kalman Sampling](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov_eks/markov_eks.md)
- [Bayesian estimates of parameters of a discrete-time Markov model using importance sampling, Markov Chain Monte Carlo and Sequential Monte Carlo with Gen.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/markov_gen/markov_gen.md)

## Equilibrium analysis

- [Steady state analysis of an SIR model with births and deaths](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_steadystate/ode_steadystate.md)
- [Bifurcation/stroboscopic plot of a sinusoidally forced ODE model using brute force simulation](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_bifurcation_bruteforce/ode_bifurcation_bruteforce.md)

## Identifiability

In conducting inference, it is important to know the extent to which parameters are identifiable from the available data.

- [Identifiability analysis of an ODE model using StructuralIdentifiability.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_identifiability/ode_identifiability.md)

## Uncertainty

Incorporating uncertainty in its many forms is important for using models to make decisions.

### Probabilistic integration

When solving continuous-time models like ODEs, the discretization can lead to numerical errors. Probabilistic integration treats this error as a statistical problem to capture the uncertainty in the model outputs generated using the solver.

- [Probabilistic integration of an ODE model using Bayesian filtering and `ProbNumDiffEq.jl`](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_probint_probnumdiffeq/ode_probint_probnumdiffeq.md)
- [Probabilistic integration of an ODE model by converting to a SDE using `DiffEqUncertainty.jl` (can also handle SDEs and DDEs)](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_probint_diffequncertainty/ode_probint_diffequncertainty.md)

### Global sensitivity

- [Global sensitivity analysis of an ODE model using Latin hypercube sampling](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_global_sensitivity/ode_global_sensitivity.md)
- [Global sensitivity analysis of an ODE model using multiple algorithms from GlobalSensitivity.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_global_sensitivity2/ode_global_sensitivity2.md)
- [Uncertainty propagation of an ODE model using MonteCarloMeasurements.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_montecarlomeasurements/ode_montecarlomeasurements.md)
- [Uncertainty propagation of an ODE model using the Koopman expectation and DiffEqUncertainty.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_koopman/ode_koopman.md)
- [Uncertainty analysis of an ODE model with an uncertain input and an uncertain output using Bayesian melding](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_bayesian_melding_1d/ode_bayesian_melding_1d.md)

### Local sensitivity

- [Local sensitivity of an ODE model using Zygote.jl and DiffEqSensitivity.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_local_sensitivity/ode_local_sensitivity.md)

### Likelihood intervals

- [Full likelihood intervals using ProfileLikelihood.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_likelihoodintervals/ode_likelihoodintervals.md)
- [Profile likelihood using ProfileLikelihood.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_profilelikelihood/ode_profilelikelihood.md)
- [Profile likelihood using LikelihoodProfiler.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_likelihoodprofiler/ode_likelihoodprofiler.md)

### Surrogate models

- [Surrogate models (single input/single output) of an ODE model using Surrogates.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_surrogate_1d_1d/ode_surrogate_1d_1d.md)
- [Gaussian process surrogate model (two inputs/one output) of an ODE model using the Python package `mogp-emulator`, with an example of history matching](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_mogp/ode_mogp.md)
- [Bayes linear surrogate model (two inputs/one output) of an ODE model using the R package `hmer`, with an example of history matching](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_hmer/ode_hmer.md)

### Flexible models

- [Data-driven models of an ODE using DataDrivenDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ddeq/ode_ddeq.md)
- [A neural ODE, which replaces all the derivatives of the model with a neural network](https://github.com/epirecipes/sir-julia/blob/master/markdown/node/node.md)
- [A universal ODE, modeling force of infection using a neural network](https://github.com/epirecipes/sir-julia/blob/master/markdown/ude/ude.md)
- [A partially specified ODE, modeling force of infection using a basis](https://github.com/epirecipes/sir-julia/blob/master/markdown/psm/psm.md)

## Extensions

- Fixed (rather than exponential) distribution of infectious period:
    - [Using a delay differential equation and DelayDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/dde/dde.md)
    - [Using a fixed delay in a jump system, making use of the integrator](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_delay/jump_process_delay.md)
    - [Using a gamma-distributed delay in a jump system using DelaySSAToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/jump_process_delayssatoolkit/jump_process_delayssatoolkit.md)
    - [Using a stochastic delay differential equation in StochasticDelayDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/sdde/sdde.md)
    - [Using a fixed delay in a discrete event simulation using SimJulia (see end of file)](https://github.com/epirecipes/sir-julia/blob/master/markdown/des/des.md)
- An Erlang distribution for the infectious period using the method of stages:
  - [Using AlgebraicDynamics.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_algebraicdynamics/ode_algebraicdynamics.md)
  - [Using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_stages/ode_stages.md)
- Multigroup models
  - [A multigroup ODE model using ModelingToolkit.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_multigroup/ode_multigroup.md)
- Fractional differential equations
  - [Using FdeSolver.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_fdesolver/ode_fdesolver.md)
  - [Using FractionalDiffEq.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_fractionaldiffeq/ode_fractionaldiffeq.md)

## Interoperability with other languages

While this repository is mainly about Julia, it is also possible to use Julia to call code written in other languages. Here are some examples of how to define the vector field of an ODE in C, Python, and R.

### Using ccall

Many languages can compile to shared libraries that can be accessed via `ccall`. Here are examples of how to define the vector field of an ODE in various languages, and call it using `ccall`.

- [ODE with derivatives in C](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ccall/ode_ccall.md)
- [ODE with derivatives in Fortran 90](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ccall_f90/ode_ccall_f90.md)
- [ODE with derivatives in Rust](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ccall_rust/ode_ccall_rust.md)
- [ODE with derivatives in FreePascal](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ccall_fpc/ode_ccall_fpc.md)
- [ODE with derivatives in Zig](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ccall_zig/ode_ccall_zig.md)
- [ODE with derivatives in Nim](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ccall_nim/ode_ccall_nim.md)

### Using Python and R

- [ODE with derivatives in Python, accessed via PythonCall.jl, also demonstrating Python to Julia code conversion using `modelingtoolkitize`](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_pythoncall/ode_pythoncall.md)
- [ODE with derivatives in R, accessed via RCall.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_rcall/ode_rcall.md)
- [ODE with derivatives defined using the `odin` R package, accessed via RCall.jl](https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_rcall_odin/ode_rcall_odin.md)

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

Plans for new examples are typically posted on the [Issues page](https://github.com/epirecipes/sir-julia/issues).

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


Change to the root directory of the repository and run the following from within Julia; you will need [Weave.jl](https://github.com/JunoLab/Weave.jl) and any dependencies from the tutorial.

```julia
include("build.jl")
weave_all() # or e.g. weave_folder("abm") for an individual tutorial
```

If additional packages are added, then these need to be added to `build_project_toml.jl`, which when run, will regenerate `Project.toml`.

## Acknowledgements

Examples use the following libraries (see the `Project.toml` file for a full list of dependencies):

- The [`DifferentialEquations.jl`](https://github.com/SciML/DifferentialEquations.jl) ecosystem for many of the examples
- [`SimJulia`](https://github.com/BenLauwens/SimJulia.jl) for discrete event simulations
- [`Agents.jl`](https://github.com/JuliaDynamics/Agents.jl) for agent-based models
- [`Gillespie.jl`](https://github.com/sdwfrost/Gillespie.jl) for the Doob-Gillespie process
- [`Petri.jl`](https://github.com/mehalter/Petri.jl) for the Petri net models
- [`AlgebraicPetri.jl`](https://github.com/AlgebraicJulia/AlgebraicPetri.jl) for a category theory based modeling framework for creating Petri net models
- [`Turing.jl`](https://turing.ml) for inference using probabilistic programs
- [`NestedSamplers.jl`](https://github.com/TuringLang/NestedSamplers.jl) for nested sampling
- [`GpABC`](https://github.com/tanhevg/GpABC.jl) for inference using Approximate Bayesian Computation
- [`Soss.jl`](https://github.com/cscherrer/Soss.jl) for Markov models
- [`MomentClosure.jl`](https://github.com/augustinas1/MomentClosure.jl/) for moment closure
- [`Bridge.jl`](https://github.com/mschauer/Bridge.jl) for stochastic differential equations

Parts of the code were taken from @ChrisRackauckas [`DiffEqTutorials`](https://github.com/SciML/DiffEqTutorials.jl), which comes highly recommended.
