# Function map using DynamicalSystems.jl
Simon Frost (@sdwfrost), 2020-04-27

## Introduction

The function map approach taken here is:

- Deterministic
- Discrete in time
- Continuous in state

This tutorial uses `DynamicalSystems.jl` to define a function map model.

## Libraries

```julia
using DynamicalSystems
using DataFrames
using StatsPlots
using BenchmarkTools
```




## Utility functions

To assist in comparison with the continuous time models, we define a function that takes a constant rate, `r`, over a timespan, `t`, and converts it to a proportion.

```julia
@inline function rate_to_proportion(r,t)
    1-exp(-r*t)
end;
```




## Transitions

We define a function that takes the 'old' state variables, `u`, and writes the 'new' state variables into `du`. Note that the timestep, `δt`, is passed as an explicit parameter.

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

```julia
δt = 0.1
nsteps = 400
tmax = nsteps*δt
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
ds = DiscreteDynamicalSystem(sir_map!, u0, p, t0 = 0)
```

```
3-dimensional discrete dynamical system
 state:       [990.0, 10.0, 0.0]
 rule f:      sir_map!
 in-place?    true
 jacobian:    ForwardDiff
 parameters:  [0.05, 10.0, 0.25, 0.1]
```



```julia
sol = trajectory(ds,nsteps)
```

```
3-dimensional Dataset{Float64} with 401 points
 990.0    10.0       0.0
 989.505  10.248     0.246901
 988.998  10.5018    0.499924
 988.479  10.7617    0.759216
 987.947  11.0278    1.02492
 987.403  11.3001    1.2972
 986.845  11.5788    1.5762
 986.274  11.8641    1.86208
 985.689  12.1561    2.15501
 985.09   12.4548    2.45514
   ⋮               
 204.289  16.4088  779.302
 204.122  16.1712  779.707
 203.957  15.9369  780.106
 203.794  15.7059  780.5
 203.634  15.4781  780.887
 203.477  15.2535  781.27
 203.322  15.032   781.646
 203.169  14.8136  782.017
 203.019  14.5983  782.383
```





## Post-processing

We can convert the output (a `DataSet`) to a dataframe for convenience.

```julia
df = DataFrame(Matrix(sol))
df[!,:t] = t;
```




## Plotting

We can now plot the results.

```julia
@df df plot(:t,
    [:x1, :x2, :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")
```

![](figures/function_map_dynamicalsystems_10_1.png)



## Benchmarking

```julia
@benchmark trajectory(ds,nsteps)
```

```
BenchmarkTools.Trial: 
  memory estimate:  9.95 KiB
  allocs estimate:  6
  --------------
  minimum time:     16.086 μs (0.00% GC)
  median time:      18.550 μs (0.00% GC)
  mean time:        20.008 μs (2.04% GC)
  maximum time:     4.109 ms (99.42% GC)
  --------------
  samples:          10000
  evals/sample:     1
```


