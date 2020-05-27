# Markov model
Simon Frost (@sdwfrost), 2020-04-27

## Introduction

The Markov model approach taken here is:

- Stochastic
- Discrete in time
- Discrete in state

## Libraries

````julia
using DifferentialEquations
using SimpleDiffEq
using Distributions
using Random
using DataFrames
using StatsPlots
using BenchmarkTools
````





## Utility functions

````julia
@inline function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;
````


````
rate_to_proportion (generic function with 1 method)
````





## Transitions

````julia
function sir_markov!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ,δt) = p
    N = S+I+R
    ifrac = rate_to_proportion(β*c*I/N,δt)
    rfrac = rate_to_proportion(γ,δt)
    infection=rand(Binomial(S,ifrac))
    recovery=rand(Binomial(I,rfrac))
    @inbounds begin
        du[1] = S-infection
        du[2] = I+infection-recovery
        du[3] = R+recovery
    end
    nothing
end;
````


````
sir_markov! (generic function with 1 method)
````





## Time domain

Note that even though we're using fixed time steps, `DifferentialEquations.jl` complains if I pass integer timespans, so I set the timespan to be `Float64`.

````julia
δt = 0.1
nsteps = 400
tmax = nsteps*δt
tspan = (0.0,nsteps)
t = 0.0:δt:tmax;
````


````
0.0:0.1:40.0
````





## Initial conditions

````julia
u0 = [990,10,0]; # S,I,R
````


````
3-element Array{Int64,1}:
 990
  10
   0
````





## Parameter values

````julia
p = [0.05,10.0,0.25,δt]; # β,c,γ,δt
````


````
4-element Array{Float64,1}:
  0.05
 10.0
  0.25
  0.1
````





## Random number seed

````julia
Random.seed!(1234);
````


````
Random.MersenneTwister(UInt32[0x000004d2], Random.DSFMT.DSFMT_state(Int32[-
1393240018, 1073611148, 45497681, 1072875908, 436273599, 1073674613, -20437
16458, 1073445557, -254908435, 1072827086  …  -599655111, 1073144102, 36765
5457, 1072985259, -1278750689, 1018350124, -597141475, 249849711, 382, 0]),
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0], UInt128[0x00000000000000000000000000000000, 
0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x0
0000000000000000000000000000000, 0x00000000000000000000000000000000, 0x0000
0000000000000000000000000000, 0x00000000000000000000000000000000, 0x0000000
0000000000000000000000000, 0x00000000000000000000000000000000, 0x0000000000
0000000000000000000000  …  0x00000000000000000000000000000000, 0x0000000000
0000000000000000000000, 0x00000000000000000000000000000000, 0x0000000000000
0000000000000000000, 0x00000000000000000000000000000000, 0x0000000000000000
0000000000000000, 0x00000000000000000000000000000000, 0x0000000000000000000
0000000000000, 0x00000000000000000000000000000000, 0x0000000000000000000000
0000000000], 1002, 0)
````





## Running the model

````julia
prob_markov = DiscreteProblem(sir_markov!,u0,tspan,p)
````


````
DiscreteProblem with uType Array{Int64,1} and tType Float64. In-place: true
timespan: (0.0, 400.0)
u0: [990, 10, 0]
````



````julia
sol_markov = solve(prob_markov,solver=FunctionMap());
````


````
retcode: Success
Interpolation: left-endpoint piecewise constant
t: 401-element Array{Float64,1}:
   0.0
   1.0
   2.0
   3.0
   4.0
   5.0
   6.0
   7.0
   8.0
   9.0
   ⋮
 392.0
 393.0
 394.0
 395.0
 396.0
 397.0
 398.0
 399.0
 400.0
u: 401-element Array{Array{Int64,1},1}:
 [990, 10, 0]
 [990, 10, 0]
 [990, 10, 0]
 [990, 10, 0]
 [990, 10, 0]
 [989, 11, 0]
 [987, 13, 0]
 [985, 14, 1]
 [984, 15, 1]
 [984, 13, 3]
 ⋮
 [179, 15, 806]
 [179, 15, 806]
 [179, 15, 806]
 [179, 14, 807]
 [179, 11, 810]
 [179, 11, 810]
 [179, 9, 812]
 [179, 9, 812]
 [178, 10, 812]
````





## Post-processing

We can convert the output to a dataframe for convenience.

````julia
df_markov = DataFrame(sol_markov')
df_markov[!,:t] = t;
````


````
0.0:0.1:40.0
````





## Plotting

We can now plot the results.

````julia
@df df_markov plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")
````


![](figures/markov_11_1.png)



## Benchmarking

````julia
@benchmark solve(prob_markov,solver=FunctionMap)
````


````
BenchmarkTools.Trial: 
  memory estimate:  58.73 KiB
  allocs estimate:  472
  --------------
  minimum time:     118.169 μs (0.00% GC)
  median time:      235.112 μs (0.00% GC)
  mean time:        265.145 μs (2.50% GC)
  maximum time:     14.864 ms (97.62% GC)
  --------------
  samples:          10000
  evals/sample:     1
````




## Appendix
### Computer Information
```
Julia Version 1.4.1
Commit 381693d3df* (2020-04-14 17:20 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, icelake-client)
Environment:
  JULIA_NUM_THREADS = 4

```

### Package Information

```
Status `~/.julia/environments/v1.4/Project.toml`
[46ada45e-f475-11e8-01d0-f70cc89e6671] Agents 3.1.0
[f5f396d3-230c-5e07-80e6-9fadf06146cc] ApproxBayes 0.3.2
[c52e3926-4ff0-5f6e-af25-54175e0327b1] Atom 0.12.11
[6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf] BenchmarkTools 0.5.0
[a134a8b2-14d6-55f6-9291-3336d3ab0209] BlackBoxOptim 0.5.0
[2445eb08-9709-466a-b3fc-47e12bd697a2] DataDrivenDiffEq 0.3.1
[a93c6f00-e57d-5684-b7b6-d8193f3e46c0] DataFrames 0.21.1
[ebbdde9d-f333-5424-9be2-dbf1e9acfb5e] DiffEqBayes 2.14.1
[459566f4-90b8-5000-8ac3-15dfb0a30def] DiffEqCallbacks 2.13.2
[aae7a2af-3d4f-5e19-a356-7da93b79d9d0] DiffEqFlux 1.10.3
[c894b116-72e5-5b58-be3c-e6d8d4ac2b12] DiffEqJump 6.7.5
[1130ab10-4a5a-5621-a13d-e4788d82bd4c] DiffEqParamEstim 1.14.1
[41bf760c-e81c-5289-8e54-58b1f1f8abe2] DiffEqSensitivity 6.17.0
[0c46a032-eb83-5123-abaf-570d42b7fbaa] DifferentialEquations 6.14.0
[b4f34e82-e78d-54a5-968a-f98e89d6e8f7] Distances 0.8.2
[31c24e10-a181-5473-b8eb-7969acd0382f] Distributions 0.23.2
[634d3b9d-ee7a-5ddf-bec9-22491ea816e1] DrWatson 1.13.0
[587475ba-b771-5e3f-ad9e-33799f191a9c] Flux 0.10.5
[28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71] GR 0.49.1
[523d8e89-b243-5607-941c-87d699ea6713] Gillespie 0.1.0
[e850a1a4-d859-11e8-3d54-a195e6d045d3] GpABC 0.0.1
[7073ff75-c697-5162-941a-fcdaad2a7d2a] IJulia 1.21.2
[4076af6c-e467-56ae-b986-b466b2749572] JuMP 0.21.2
[e5e0dc1b-0480-54bc-9374-aad01c23163d] Juno 0.8.2
[093fc24a-ae57-5d10-9952-331d41423f4d] LightGraphs 1.3.3
[1914dd2f-81c6-5fcd-8719-6d5c9610ff09] MacroTools 0.5.5
[961ee093-0014-501f-94e3-6117800e7a78] ModelingToolkit 3.6.4
[76087f3c-5699-56af-9a33-bf431cd00edd] NLopt 0.6.0
[429524aa-4258-5aef-a3af-852621145aeb] Optim 0.20.6
[1dea7af3-3e70-54e6-95c3-0bf5283fa5ed] OrdinaryDiffEq 5.38.2
[91a5bcdd-55d7-5caf-9e0b-520d859cae80] Plots 1.3.4
[428bdadb-6287-5aa5-874b-9969638295fd] SimJulia 0.8.0
[05bca326-078c-5bf0-a5bf-ce7c7982d7fd] SimpleDiffEq 1.1.0
[276daf66-3868-5448-9aa4-cd146d93841b] SpecialFunctions 0.10.3
[f3b207a7-027a-5e70-b257-86293d7955fd] StatsPlots 0.14.6
[789caeaf-c7a9-5a7d-9973-96adeb23e2a0] StochasticDiffEq 6.23.0
[92b13dbe-c966-51a2-8445-caca9f8a7d42] TaylorIntegration 0.8.3
[fce5fe82-541a-59a6-adf8-730c64b5f9a0] Turing 0.13.0
[44d3d7a6-8a23-5bf8-98c5-b353f8df5ec9] Weave 0.10.2
[e88e6eb3-aa80-5325-afca-941959d7151f] Zygote 0.4.20
```
