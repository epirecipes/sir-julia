# Ordinary differential equation model with the vector field defined in R
Simon Frost (@sdwfrost), 2024-06-03

## Introduction

While Julia is a high-level language, it is possible to define the vector field for an ordinary differential equation (ODE) in another language and call it from Julia. This can be useful if the vector field is already defined in R, for example, in another codebase. We use the `RCall` library to interface Julia with R.

## Libraries

```julia
using OrdinaryDiffEq
using RCall
using Plots
using BenchmarkTools
```




## Transitions

We define the vector field in R using an out-of-place definition; R passes arguments by value rather than by reference, so this approach is necessary.

```julia
R"""
sir_ode_op_r <- function(u,p,t){
    S <- u[1]
    I <- u[2]
    R <- u[3]
    N <- S+I+R
    beta <- p[1]
    cee <- p[2]
    gamma <- p[3]
    dS <- -beta*cee*I/N*S
    dI <- beta*cee*I/N*S - gamma*I
    dR <- gamma*I
    return(c(dS,dI,dR))
}
""";
```




We can then wrap the R function in a Julia function, converting the output to an `Array`.

```julia
function sir_ode_op_jl(u,p,t)
    robj = rcall(:sir_ode_op_r, u, p, t)
    return convert(Array,robj)
end;
```




We can then proceed to solve the ODE using the `sir_ode_op_jl` function as we would if the vector field were defined in Julia.

```julia
δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ
```




To ensure that the out-of-place version works, we specify `ODEProblem{false}`.

```julia
prob_ode_op = ODEProblem{false}(sir_ode_op_jl, u0, tspan, p)
sol_ode_op = solve(prob_ode_op, Tsit5(), dt = δt)
plot(sol_ode_op, labels=["S" "I" "R"], lw = 2, xlabel = "Time", ylabel = "Number")
```

![](figures/ode_rcall_5_1.png)



## Benchmarking

```julia
@benchmark solve(prob_ode_op, Tsit5(), dt = δt)
```

```
BenchmarkTools.Trial: 5320 samples with 1 evaluation.
 Range (min … max):  776.709 μs …  21.131 ms  ┊ GC (min … max): 0.00% … 0.0
0%
 Time  (median):     843.896 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   922.618 μs ± 823.581 μs  ┊ GC (mean ± σ):  4.67% ± 5.3
9%

  ▆▄ ▃█                                                          
  ██▆██▇▆▅▄▄▃▂▂▂▂▂▂▂▂▁▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▃
  777 μs           Histogram: frequency by time         1.81 ms <

 Memory estimate: 244.73 KiB, allocs estimate: 6563.
```





### Julia out-of-place version

We can compare the performance of the R-based ODE with the Julia-based ODE.

```julia
function sir_ode_op_julia(u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    dS = -β*c*I/N*S
    dI = β*c*I/N*S - γ*I
    dR = γ*I
    [dS,dI,dR]
end
prob_ode_julia = ODEProblem(sir_ode_op_julia, u0, tspan, p)
sol_ode_julia = solve(prob_ode_julia, Tsit5(), dt = δt)
@benchmark solve(prob_ode_julia, Tsit5(), dt = δt)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  34.125 μs …  10.218 ms  ┊ GC (min … max):  0.00% … 98.
75%
 Time  (median):     37.583 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   46.861 μs ± 209.155 μs  ┊ GC (mean ± σ):  17.52% ±  4.
37%

     ▄██▅▁                                                      
  ▂▃▆█████▆▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂▂▂▂▂▂▂▂▁▂▂▂▂▂ ▃
  34.1 μs         Histogram: frequency by time         71.7 μs <

 Memory estimate: 88.52 KiB, allocs estimate: 1113.
```





On my machine, the Julia code runs 20 times faster than the R code; this reflects both the slower R code plus the overhead of the foreign function calls.

At the time of writing, `modelingtoolkitize` does not work with R functions, unlike the Python version of this notebook.