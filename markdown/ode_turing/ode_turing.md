# Ordinary differential equation model with inference using Turing.jl
Simon Frost (@sdwfrost), 2020-05-27

## Introduction

In this notebook, we try to infer the parameter values from a simulated dataset using [Turing.jl](https://turing.ml).

## Libraries

```julia
using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using Turing
using DataFrames
using StatsPlots
```




## The model

The following is a standard SIR model, where we keep track of the cumulative number of infected individuals, `C`.

```julia
function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end;
```


```julia
tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax
u0 = [990.0,10.0,0.0,0.0] # S,I.R,C
p = [0.05,10.0,0.25]; # β,c,γ
```


```julia
prob_ode = ODEProblem(sir_ode!,u0,tspan,p);
```


```julia
sol_ode = solve(prob_ode,
            Tsit5(),
            saveat = 1.0);
```




## Generating data

The data are assumed to be of daily new cases, which we can obtain from the cumulative number, `C`.

```julia
C = Array(sol_ode)[4,:] # Cumulative cases
X = C[2:end] - C[1:(end-1)];
```




We generate some random Poisson noise for the measurement error to generate the observations, `Y`.

```julia
Random.seed!(1234)
Y = rand.(Poisson.(X));
```


```julia
bar(obstimes,Y,legend=false)
plot!(obstimes,X,legend=false)
```

![](figures/ode_turing_8_1.png)



## Model specification using Turing

This model estimates the initial proportion of the population that is infected, `i₀`, and the infection probability, `β`, assuming uniform priors on each, with the remaining parameters fixed.

```julia
@model bayes_sir(y) = begin
  # Calculate number of timepoints
  l = length(y)
  i₀  ~ Uniform(0.0,1.0)
  β ~ Uniform(0.0,1.0)
  I = i₀*1000.0
  u0=[1000.0-I,I,0.0,0.0]
  p=[β,10.0,0.25]
  tspan = (0.0,float(l))
  prob = ODEProblem(sir_ode!,
          u0,
          tspan,
          p)
  sol = solve(prob,
              Tsit5(),
              saveat = 1.0)
  sol_C = Array(sol)[4,:] # Cumulative cases
  sol_X = sol_C[2:end] - sol_C[1:(end-1)]
  l = length(y)
  for i in 1:l
    y[i] ~ Poisson(sol_X[i])
  end
end;
```




### Fit using NUTS

The following fits the model using the No U-Turn Sampler.

```julia
ode_nuts = sample(bayes_sir(Y),NUTS(0.65),10000);
```




The `describe` function displays some summary statistics of the output.

```julia
describe(ode_nuts)
```

```
2-element Array{MCMCChains.ChainDataFrame,1}:
 Summary Statistics (2 x 7)
 Quantiles (2 x 6)
```



```julia
plot(ode_nuts)
```

![](figures/ode_turing_12_1.png)



### Further plotting

The MCMC chains can be converted into a `DataFrame` for further plotting.

```julia
posterior = DataFrame(ode_nuts);
```


```julia
histogram2d(posterior[!,:β],posterior[!,:i₀],
                bins=80,
                xlabel="β",
                ylab="i₀",
                ylim=[0.006,0.016],
                xlim=[0.045,0.055],
                legend=false)
plot!([0.05,0.05],[0.0,0.01])
plot!([0.0,0.05],[0.01,0.01])
```

![](figures/ode_turing_14_1.png)



### Generate predictions

The following code generates predicted dynamics by sampling parameter values from the posterior distribution and running the model.

```julia
function predict(y,chain)
    # Length of data
    l = length(y)
    # Length of chain
    m = length(chain)
    # Choose random
    idx = sample(1:m)
    i₀ = chain[:i₀][idx]
    β = chain[:β][idx]
    I = i₀*1000.0
    u0=[1000.0-I,I,0.0,0.0]
    p=[β,10.0,0.25]
    tspan = (0.0,float(l))
    prob = ODEProblem(sir_ode!,
            u0,
            tspan,
            p)
    sol = solve(prob,
                Tsit5(),
                saveat = 1.0)
    out = Array(sol)
    sol_X = [0.0; out[4,2:end] - out[4,1:(end-1)]]
    hcat(sol_ode.t,out',sol_X)
end;
```




Here is a plot of ten samples of the posterior for the number of daily cases against the simulated data.

```julia
Xp = []
for i in 1:10
    pred = predict(Y,ode_nuts)
    push!(Xp,pred[2:end,6])
end
```


```julia
scatter(obstimes,Y,legend=false)
plot!(obstimes,Xp,legend=false)
```

![](figures/ode_turing_17_1.png)
