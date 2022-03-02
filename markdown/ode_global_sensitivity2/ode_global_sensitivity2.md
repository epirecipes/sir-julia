# Global sensitivity analysis applied to ordinary differential equation model
Simon Frost (@sdwfrost), 2022-03-01

## Introduction

The classical ODE version of the SIR model is:

- Deterministic
- Continuous in time
- Continuous in state

This tutorial uses tools from the `GlobalSensitivity.jl` package to explore sensitivity of the peak number of infected individuals, the timing of the peak and the final size of the epidemic to changes in parameter values.

## Libraries

```julia
using OrdinaryDiffEq
using DiffEqCallbacks
using GlobalSensitivity
using Distributions
using Plots
```




## Transitions

The following function provides the derivatives of the model, which it changes in-place. State variables and parameters are unpacked from `u` and `p`.

```julia
function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*c*I/N*S
        du[2] = β*c*I/N*S - γ*I
        du[3] = γ*I
    end
    nothing
end;
```




## Time domain

We set the timespan for simulations, `tspan`, initial conditions, `u0`, and parameter values, `p`. We will set the maximum time to be high, as we will be using a callback in order to stop the integration early.

```julia
tmax = 10000.0
tspan = (0.0,tmax);
```




## Callbacks

If we just wanted the final size, we could use a `SteadyStateProblem` with the `DynamicSS` solver. To get access to the entire solution, we can use a callback instead to stop the simulation when it reaches a steady state.

```julia
cb_ss = TerminateSteadyState();
```




## Initial conditions and parameter values

We first set fixed parameters, in this case, the total population size, `N`. In addition, in order to define an `ODEProblem`, we also need a default set of initial conditions, `u`, and parameter values, `p`.

```julia
N = 1000.0;
u0 = [990.0,10.0,0.0];
p = [0.05,10.0,0.25]; # β,c,γ
```




We specify lower (`lb`) and upper (`ub`) bounds for each parameter.

```julia
n_samples = 1000 # Number of samples
# Parameters are β, c, γ, I₀
lb = [0.01, 5.0, 0.1, 1.0]
ub = [0.1, 20.0, 1.0, 50.0]
n_params = 4;
```




## Running the model

```julia
prob_ode = ODEProblem(sir_ode!,u0,tspan,p);
```




We will consider three summary statistics of the simulation for sensitivity analysis:

1. The peak number of infected individuals, `I`.
2. The time at which the peak is reached.
3. The final size of the epidemic (as `R(0)=0`, this will be `R(t_stop)` where `t_stop` is the time at which the steady state is reached).

## Serial

In the serial implementation, we write a function that takes a `Vector` of parameters and initial conditions and returns a `Vector` of outputs.

```julia
f1 = function(pu0)
  p = pu0[1:3]
  I0 = pu0[4]
  u0 = [N-I0,I0,0.0]
  prob = remake(prob_ode;p=p,u=u0)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  [maximum(sol[2,:]), sol.t[argmax(sol[2,:])], sol[end][3]]
end;
```




### Morris method

```julia
m_morris = gsa(f1, Morris(num_trajectory=n_samples), [[lb[i],ub[i]] for i in 1:n_params]);
```


```julia
m_morris.means
```

```
3×4 Matrix{Float64}:
 2406.0     12.1933    -499.648   0.0
   60.5027   0.253414   -23.4177  0.0
 7817.26    34.0486    -944.808   0.0
```



```julia
m_morris.variances
```

```
3×4 Matrix{Float64}:
 8.89626e6   190.175       4.6575e5   0.0
 1.27243e6    27.9205  25137.9        0.0
 1.09444e8  1767.91        1.26193e6  0.0
```





### Sobol

```julia
m_sobol = gsa(f1, Sobol(), [[lb[i],ub[i]] for i in 1:n_params],N=n_samples);
```


```julia
m_sobol.ST
```

```
3×4 Matrix{Float64}:
 0.365555  0.203337  0.673186  0.0
 0.524253  0.327364  0.900231  0.0
 0.446984  0.255379  0.513369  0.0
```



```julia
m_sobol.S1
```

```
3×4 Matrix{Float64}:
  0.184066  0.110684   0.45841   0.0
 -0.010233  0.0125078  0.296178  0.0
  0.3003    0.144473   0.358434  0.0
```





### Regression

```julia
m_regression = gsa(f1, RegressionGSA(rank=true), [[lb[i],ub[i]] for i in 1:n_params]; samples = n_samples);
```


```julia
m_regression.partial_correlation
```

```
3×4 Matrix{Float64}:
  0.0221537   0.0330808  -0.429725  -0.0217352
 -0.173664   -0.163188   -0.305592  -0.0386295
  0.605671    0.516314   -0.235003   0.0309434
```



```julia
m_regression.partial_rank_correlation
```

```
3×4 Matrix{Float64}:
  0.0272697  -0.029168    -0.0158028  -0.0313885
 -0.0160025   0.015693     0.0205675   0.0633896
 -0.0030518  -0.00837281   0.0063698   0.0160815
```





### eFAST

```julia
m_efast = gsa(f1, eFAST(), [[lb[i],ub[i]] for i in 1:n_params]; n = n_samples);
```


```julia
m_efast.ST
```

```
3×4 Matrix{Float64}:
 0.3654    0.213928  0.665651  0.0138733
 0.633119  0.464074  0.960826  0.160137
 0.461577  0.264273  0.51549   0.0068332
```



```julia
m_efast.S1
```

```
3×4 Matrix{Float64}:
 0.195022   0.109223    0.480149  5.10784e-6
 0.0354833  0.00489495  0.31963   0.000524032
 0.312645   0.155381    0.379897  3.8251e-5
```





## Running the model in parallel

To run the above in parallel, we pass a `Matrix` of parameter values and use `EnsembleProblem` internally in order to run different parameter sets in parallel, returning a `Matrix` of outputs. The function below uses threads to parallelize, but it can easily be adapted to other modes of parallelism.

```julia
pf1 = function (pu0)
  p = pu0[1:3,:]
  I0 = pu0[4,:]
  prob_func(prob,i,repeat) = remake(prob;p=p[:,i],u=[N-I0[i],I0[i],0.0])
  ensemble_prob = EnsembleProblem(prob_ode,prob_func=prob_func)
  sol = solve(ensemble_prob,ROS34PW3(),EnsembleThreads();trajectories=size(p,2))
  out = zeros(3,size(p,2))
  for i in 1:size(p,2)
    out[1,i] = maximum(sol[i][2,:])
    out[2,i] = sol[i].t[argmax(sol[i][2,:])]
    out[3,i] = sol[i][end][3]
  end
  out
end;
```




We then pass the keyword `batch=true` to `gsa`; here is an example for the `eFAST` method.

```julia
m_efast_parallel = gsa(pf1, eFAST(), [[lb[i],ub[i]] for i in 1:n_params]; n = n_samples, batch = true);
```

