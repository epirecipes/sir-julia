# Function map using DiscreteProblem and FunctionMap
Simon Frost (@sdwfrost), 2020-04-27

## Introduction

The function map approach taken here is:

- Deterministic
- Discrete in time
- Continuous in state

This example uses `DiscreteProblem` and `FunctionMap` from `OrdinaryDiffEq.jl` to solve the SIR model in a discrete time setting.

## Libraries

```julia
using OrdinaryDiffEq
using SimpleDiffEq
using Plots
using BenchmarkTools;
```




## Utility functions

To assist in comparison with the continuous time models, we define a function that takes a constant rate, `r`, over a timespan, `t`, and converts it to a proportion.

```julia
@inline function rate_to_proportion(r,t)
    1-exp(-r*t)
end;
```




## Transitions

We define a function that takes the 'old' state variables, `u`, and writes the 'new' state variables into `du.` Note that the timestep, `δt`, is passed as an explicit parameter.

```julia
function sir_map!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ,δt) = p
    N = S+I+R
    infection = rate_to_proportion(β*c*I/N,δt)*S
    recovery = rate_to_proportion(γ,δt)*I
    @inbounds begin
        du[1] = S-infection
        du[2] = I+infection-recovery
        du[3] = R+recovery
    end
    nothing
end;
```




## Time domain

Note that even though I'm using fixed time steps, `DifferentialEquations.jl` complains if I pass integer timespans, so I set the timespan to be `Float64`.

```julia
δt = 0.1
nsteps = 400
tmax = nsteps*δt
tspan = (0.0,nsteps)
t = 0.0:δt:tmax;
```




## Initial conditions

Note that we define the state variables as floating point.

```julia
u0 = [990.0,10.0,0.0];
```




## Parameter values

```julia
p = [0.05,10.0,0.25,δt]; # β,c,γ,δt
```




## Running the model

```julia
prob_map = DiscreteProblem(sir_map!,u0,tspan,p);
```


```julia
sol_map = solve(prob_map,FunctionMap());
```




## Post-processing

```julia
# Extract state variables
t = sol_map.t
S = [u[1] for u in sol_map.u]
I = [u[2] for u in sol_map.u]
R = [u[3] for u in sol_map.u];
```




## Plotting

We can now plot the results.

```julia
plot(t,
     [S I R],
     label=["S" "I" "R"],
     xlabel="Time",
     ylabel="Number")
```

![](figures/function_map_10_1.png)



## Benchmarking

```julia
@benchmark solve(prob_map, FunctionMap())
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  37.750 μs …  17.281 ms  ┊ GC (min … max): 0.00% … 99.5
5%
 Time  (median):     42.041 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   45.951 μs ± 213.678 μs  ┊ GC (mean ± σ):  6.48% ±  1.4
1%

    ▅▆▇▅█▅▄█▅▇▅▇▄▄▃▂▂▂▁                                         
  ▂▄████████████████████▇█▇▇▅▆▅▅▄▄▄▄▄▃▄▃▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁ ▄
  37.8 μs         Histogram: frequency by time         56.3 μs <

 Memory estimate: 56.20 KiB, allocs estimate: 1238.
```





Using `FunctionMap` is much slower than using a loop to fill in the entries of a pre-allocated array, as in [another example](https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_vanilla/function_map_vanilla.md). Instead, we can use `SimpleFunctionMap` from `SimpleDiffEq.jl` to get a significant speedup.

```julia
@benchmark solve(prob_map, SimpleFunctionMap())
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  14.917 μs …  17.758 ms  ┊ GC (min … max): 0.00% … 99.7
0%
 Time  (median):     16.750 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   18.923 μs ± 177.415 μs  ┊ GC (mean ± σ):  9.36% ±  1.0
0%

     ▄▇▃▅█▃▄▃▂▃▃▁▃▃                                             
  ▁▂▄██████████████▇▇▆▅▆▅▅▅▃▄▃▃▂▃▂▂▂▁▂▂▂▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  14.9 μs         Histogram: frequency by time           24 μs <

 Memory estimate: 35.08 KiB, allocs estimate: 806.
```


