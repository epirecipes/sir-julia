# Ordinary differential equation model with inference of point estimates using optimization
Simon Frost (@sdwfrost), 2020-04-27

## Introduction

The classical ODE version of the SIR model is:

- Deterministic
- Continuous in time
- Continuous in state

In this notebook, we try to infer the parameter values from a simulated dataset.

## Libraries

```julia
using DifferentialEquations
using SimpleDiffEq
using DiffEqSensitivity
using Random
using Distributions
using DiffEqParamEstim
using Plots
```




## Transitions

The following function provides the derivatives of the model, which it changes in-place. State variables and parameters are unpacked from `u` and `p`; this incurs a slight performance hit, but makes the equations much easier to read.

A variable is included for the cumulative number of infections, $C$.

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




## Time domain

We set the timespan for simulations, `tspan`, initial conditions, `u0`, and parameter values, `p` (which are unpacked above as `[β,γ]`).

```julia
δt = 1.0
tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax;
```




## Initial conditions

```julia
u0 = [990.0,10.0,0.0,0.0]; # S,I.R,Y
```




## Parameter values

```julia
p = [0.05,10.0,0.25]; # β,c,γ
```




## Running the model

```julia
prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode,Tsit5(),saveat=δt);
```




## Generating data

The cumulative counts are extracted.

```julia
out = Array(sol_ode)
C = out[4,:];
```




The new cases per day are calculated from the cumulative counts.

```julia
X = C[2:end] .- C[1:(end-1)];
```




Although the ODE system is deterministic, we can add measurement error to the counts of new cases. Here, a Poisson distribution is used, although a negative binomial could also be used (which would introduce an additional parameter for the variance).

```julia
Random.seed!(1234);
```


```julia
Y = rand.(Poisson.(X));
```




## Using Optim.jl directly

```julia
using Optim
```




### Single parameter optimization

This function calculates the sum of squares for a single parameter fit (β). Note how the original `ODEProblem` is remade using the `remake` function. Like all the loss functions listed here, `Inf` is returned if the number of daily cases is less than or equal to zero.

```julia
function ss1(β)
    prob = remake(prob_ode,u0=[990.0,10.0,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    return(sum((X .- Y) .^2))
end;
```




Optimisation routines typically *minimise* functions, so for maximum likelihood estimates, we have to define the *negative* log-likelihood - here, for a single parameter, β.

```julia
function nll1(β)
    prob = remake(prob_ode,u0=[990.0,10.0,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -sum(logpdf.(Poisson.(X),Y))
end;
```




In this model, β is positive and (through the meaning of the parameter) bounded between 0 and 1. For point estimates, we could use constrained optimisation, or transform β to an unconstrained scale. Here is the first approach, defining the bounds and initial values for optimization.

```julia
lower1 = 0.0
upper1 = 1.0
initial_x1 = 0.1;
```




Model fit using sum of squares. The output isn't suppressed, as the output of the outcome of the optimisation, such as whether it has converged, is important.

```julia
opt1_ss = Optim.optimize(ss1,lower1,upper1)
```

```
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [0.000000, 1.000000]
 * Minimizer: 4.909134e-02
 * Minimum: 1.016203e+03
 * Iterations: 22
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16
): true
 * Objective Function Calls: 23
```





Model fit using (negative) log likelihood.

```julia
opt1_nll = Optim.optimize(nll1,lower1,upper1)
```

```
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [0.000000, 1.000000]
 * Minimizer: 4.969331e-02
 * Minimum: 1.117286e+02
 * Iterations: 24
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16
): true
 * Objective Function Calls: 25
```





### Multiparameter optimization

Multiple parameters are handled in the cost function using an array argument. Firstly, sum of squares.

```julia
function ss2(x)
    (i0,β) = x
    I = i0*1000.0
    prob = remake(prob_ode,u0=[1000.0-I,I,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    return(sum((X .- Y) .^2))
end;
```




Secondly, negative log-likelihood.

```julia
function nll2(x)
    (i0,β) = x
    I = i0*1000.0
    prob = remake(prob_ode,u0=[1000.0-I,I,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -sum(logpdf.(Poisson.(X),Y))
end;
```




Two-parameter lower and upper bounds and initial conditions.

```julia
lower2 = [0.0,0.0]
upper2 = [1.0,1.0]
initial_x2 = [0.01,0.1];
```


```julia
opt2_ss = Optim.optimize(ss2,lower2,upper2,initial_x2)
```

```
* Status: success

 * Candidate solution
    Final objective value:     1.013835e+03

 * Found with
    Algorithm:     Fminbox with L-BFGS

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 1.37e-01 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    4
    f(x) calls:    234
    ∇f(x) calls:   234
```



```julia
opt2_nll = Optim.optimize(nll2,lower2,upper2,initial_x2)
```

```
* Status: success

 * Candidate solution
    Final objective value:     1.117286e+02

 * Found with
    Algorithm:     Fminbox with L-BFGS

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 4.30e-02 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    3
    f(x) calls:    167
    ∇f(x) calls:   167
```





## Using DiffEqParamEstim

The advantage of using a framework such as DiffEqParamEstim is that a number of different frameworks can be employed easily. Firstly, the loss function is defined.

```julia
function loss_function(sol)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -sum(logpdf.(Poisson.(X),Y))
end;
```




Secondly, a function that generates the `Problem` to be solved.

```julia
prob_generator = (prob,q) -> remake(prob,
    u0=[1000.0-(q[1]*1000),q[1]*1000,0.0,0.0],
    p=[q[2],10.0,0.25]);
```




The loss function and the problem generator then get combined to build the objective function.

```julia
cost_function = build_loss_objective(prob_ode,
    Tsit5(),
    loss_function,
    saveat=δt,
    prob_generator = prob_generator,
    maxiters=100,
    verbose=false);
```




### Optim interface

The resulting cost function can be passed to `Optim.jl` as before.

```julia
opt_pe1 = Optim.optimize(cost_function,lower2,upper2,initial_x2)
```

```
* Status: success

 * Candidate solution
    Final objective value:     1.117286e+02

 * Found with
    Algorithm:     Fminbox with L-BFGS

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 4.30e-02 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    3
    f(x) calls:    167
    ∇f(x) calls:   167
```





### NLopt interface

The same function can also be passed to `NLopt.jl`. For some reason, this reaches the maximum number of evaluations.

```julia
using NLopt
opt = Opt(:LD_MMA, 2)
opt.lower_bounds = lower2
opt.upper_bounds = upper2
opt.min_objective = cost_function
opt.maxeval = 10000
(minf,minx,ret) = NLopt.optimize(opt,initial_x2)
```

```
(111.72855334375784, [0.010000860197548981, 0.04969282386672637], :MAXEVAL_
REACHED)
```





### BlackBoxOptim interface

We can also use `BlackBoxOptim.jl`.

```julia
using BlackBoxOptim
bound1 = Tuple{Float64, Float64}[(0.0,1.0),(0.0, 1.0)]
result = bboptimize(cost_function;SearchRange = bound1, MaxSteps = 1e4)
```

```
Starting optimization with optimizer BlackBoxOptim.DiffEvoOpt{BlackBoxOptim
.FitPopulation{Float64},BlackBoxOptim.RadiusLimitedSelector,BlackBoxOptim.A
daptiveDiffEvoRandBin{3},BlackBoxOptim.RandomBound{BlackBoxOptim.Continuous
RectSearchSpace}}
0.00 secs, 0 evals, 0 steps
0.50 secs, 9125 evals, 9007 steps, improv/step: 0.155 (last = 0.1548), fitn
ess=111.728553108

Optimization stopped after 10001 steps and 0.55 seconds
Termination reason: Max number of steps (10000) reached
Steps per second = 18101.00
Function evals per second = 18314.57
Improvements/step = 0.14040
Total function evaluations = 10119


Best candidate found: [0.0100005, 0.049693]

Fitness: 111.728553108

BlackBoxOptim.OptimizationResults("adaptive_de_rand_1_bin_radiuslimited", "
Max number of steps (10000) reached", 10001, 1.615909946463479e9, 0.5525109
767913818, BlackBoxOptim.DictChain{Symbol,Any}[BlackBoxOptim.DictChain{Symb
ol,Any}[Dict{Symbol,Any}(:RngSeed => 122218,:SearchRange => [(0.0, 1.0), (0
.0, 1.0)],:MaxSteps => 10000),Dict{Symbol,Any}()],Dict{Symbol,Any}(:Fitness
Scheme => BlackBoxOptim.ScalarFitnessScheme{true}(),:NumDimensions => :NotS
pecified,:PopulationSize => 50,:MaxTime => 0.0,:SearchRange => (-1.0, 1.0),
:Method => :adaptive_de_rand_1_bin_radiuslimited,:MaxNumStepsWithoutFuncEva
ls => 100,:RngSeed => 1234,:MaxFuncEvals => 0,:SaveTrace => false…)], 10119
, BlackBoxOptim.ScalarFitnessScheme{true}(), BlackBoxOptim.TopListArchiveOu
tput{Float64,Array{Float64,1}}(111.72855310778279, [0.010000493098277958, 0
.049692988027000715]), BlackBoxOptim.PopulationOptimizerOutput{BlackBoxOpti
m.FitPopulation{Float64}}(BlackBoxOptim.FitPopulation{Float64}([0.010000450
352575409 0.01000051617297052 … 0.010000415410773981 0.010000388636819235; 
0.049693052322031875 0.04969305524867424 … 0.04969298384180534 0.0496930187
2491333], NaN, [111.72855320086643, 111.72855318865552, 111.72855316901004,
 111.72855322267971, 111.72855312801056, 111.72855323083962, 111.7285531829
84, 111.72855318680475, 111.72855321835844, 111.72855315982086  …  111.7285
5320289815, 111.72855325037932, 111.728553194787, 111.72855318119758, 111.7
2855321331336, 111.72855310778279, 111.72855320830769, 111.72855325660713, 
111.72855318438637, 111.72855322250844], 0, BlackBoxOptim.Candidate{Float64
}[BlackBoxOptim.Candidate{Float64}([0.01000026658013334, 0.0496932389629908
1], 17, 111.72855321617126, BlackBoxOptim.AdaptiveDiffEvoRandBin{3}(BlackBo
xOptim.AdaptiveDiffEvoParameters(BlackBoxOptim.BimodalCauchy(Distributions.
Cauchy{Float64}(μ=0.65, σ=0.1), Distributions.Cauchy{Float64}(μ=1.0, σ=0.1)
, 0.5, false, true), BlackBoxOptim.BimodalCauchy(Distributions.Cauchy{Float
64}(μ=0.1, σ=0.1), Distributions.Cauchy{Float64}(μ=0.95, σ=0.1), 0.5, false
, true), [0.8058793348772144, 0.6115273170026571, 0.6146571995133624, 0.714
1815572058267, 0.9187455486809828, 0.5958790981424303, 0.6188311256508622, 
1.0, 1.0, 0.514201780435776  …  0.7990965471309444, 0.9928797995662134, 1.0
, 0.7831046867992633, 0.5321696504636098, 0.6482130772726755, 0.50989106040
57572, 0.5672194723184378, 0.5546750454935506, 1.0], [1.0, 0.16665499001680
187, 0.8985810529190772, 0.08255732299945209, 0.10139688043250322, 0.855416
007145057, 0.13085260018626185, 0.07936728444977068, 0.07838668225076971, 1
.0  …  0.8228759916835053, 1.0, 0.8207461502008848, 0.23317117329265827, 0.
10072238264907452, 0.5315400405644214, 0.6548047505233879, 0.66633643691050
37, 0.7093835195987694, 0.0705210407466892])), 0), BlackBoxOptim.Candidate{
Float64}([0.010000363674035216, 0.04969323896299081], 17, 111.7285534667933
9, BlackBoxOptim.AdaptiveDiffEvoRandBin{3}(BlackBoxOptim.AdaptiveDiffEvoPar
ameters(BlackBoxOptim.BimodalCauchy(Distributions.Cauchy{Float64}(μ=0.65, σ
=0.1), Distributions.Cauchy{Float64}(μ=1.0, σ=0.1), 0.5, false, true), Blac
kBoxOptim.BimodalCauchy(Distributions.Cauchy{Float64}(μ=0.1, σ=0.1), Distri
butions.Cauchy{Float64}(μ=0.95, σ=0.1), 0.5, false, true), [0.8058793348772
144, 0.6115273170026571, 0.6146571995133624, 0.7141815572058267, 0.91874554
86809828, 0.5958790981424303, 0.6188311256508622, 1.0, 1.0, 0.5142017804357
76  …  0.7990965471309444, 0.9928797995662134, 1.0, 0.7831046867992633, 0.5
321696504636098, 0.6482130772726755, 0.5098910604057572, 0.5672194723184378
, 0.5546750454935506, 1.0], [1.0, 0.16665499001680187, 0.8985810529190772, 
0.08255732299945209, 0.10139688043250322, 0.855416007145057, 0.130852600186
26185, 0.07936728444977068, 0.07838668225076971, 1.0  …  0.8228759916835053
, 1.0, 0.8207461502008848, 0.23317117329265827, 0.10072238264907452, 0.5315
400405644214, 0.6548047505233879, 0.6663364369105037, 0.7093835195987694, 0
.0705210407466892])), 0)])))
```


