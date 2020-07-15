# Ordinary differential equation model with inference of point estimates using optimization
Simon Frost (@sdwfrost), 2020-04-27

## Introduction

The classical ODE version of the SIR model is:

- Deterministic
- Continuous in time
- Continuous in state

In this notebook, we try to infer the parameter values from a simulated dataset.

## Libraries

````julia
using DifferentialEquations
using SimpleDiffEq
using DiffEqSensitivity
using Random
using Distributions
using DiffEqParamEstim
using Plots
````





## Transitions

The following function provides the derivatives of the model, which it changes in-place. State variables and parameters are unpacked from `u` and `p`; this incurs a slight performance hit, but makes the equations much easier to read.

A variable is included for the cumulative number of infections, $C$.

````julia
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
````


````
sir_ode! (generic function with 1 method)
````





## Time domain

We set the timespan for simulations, `tspan`, initial conditions, `u0`, and parameter values, `p` (which are unpacked above as `[β,γ]`).

````julia
δt = 1.0
tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax;
````


````
1.0:1.0:40.0
````





## Initial conditions

````julia
u0 = [990.0,10.0,0.0,0.0]; # S,I.R,Y
````


````
4-element Array{Float64,1}:
 990.0
  10.0
   0.0
   0.0
````





## Parameter values

````julia
p = [0.05,10.0,0.25]; # β,c,γ
````


````
3-element Array{Float64,1}:
  0.05
 10.0
  0.25
````





## Running the model

````julia
prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode,Tsit5(),saveat=δt);
````


````
retcode: Success
Interpolation: 1st order linear
t: 41-element Array{Float64,1}:
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
 32.0
 33.0
 34.0
 35.0
 36.0
 37.0
 38.0
 39.0
 40.0
u: 41-element Array{Array{Float64,1},1}:
 [990.0, 10.0, 0.0, 0.0]
 [984.4093729820466, 12.759075298665039, 2.83155171928832, 5.59062701795335
75]
 [977.3332026510055, 16.22814554943427, 6.438651799560168, 12.6667973489944
34]
 [968.4242204222029, 20.558426973853646, 11.017352603943419, 21.57577957779
7064]
 [957.2822747333526, 25.91440777703591, 16.80331748961151, 32.7177252666474
1]
 [943.4628419127473, 32.463172502357864, 24.073985584894846, 46.53715808725
271]
 [926.497065600148, 40.35593403853616, 33.14700036131587, 63.50293439985202
]
 [905.9269551962524, 49.699931631668576, 44.37311317207904, 84.073044803747
64]
 [881.3585392131616, 60.52139921718387, 58.12006156965444, 108.641460786838
32]
 [852.5274163604042, 72.72290972680412, 74.74967391279168, 137.472583639595
8]
 ⋮
 [233.6919415512116, 44.46580488532527, 721.8422535634633, 756.308058448788
4]
 [228.88046764765463, 38.871629484210395, 732.247902868135, 761.11953235234
55]
 [224.76186839108456, 33.90671598976695, 741.3314156191486, 765.23813160891
54]
 [221.23300305457357, 29.522228842544973, 749.2447681028816, 768.7669969454
265]
 [218.20665229931438, 25.662350534147386, 756.1309971665382, 771.7933477006
857]
 [215.6107449890835, 22.274515459755385, 762.1147395511612, 774.38925501091
65]
 [213.3843905024685, 19.310740281779456, 767.3048692157521, 776.61560949753
15]
 [211.47427574879592, 16.72457203571691, 771.8011522154873, 778.52572425120
41]
 [209.83434778204648, 14.47208748405588, 775.6935647338977, 780.16565221795
35]
````





## Generating data

The cumulative counts are extracted.

````julia
out = Array(sol_ode)
C = out[4,:];
````


````
41-element Array{Float64,1}:
   0.0
   5.5906270179533575
  12.666797348994434
  21.575779577797064
  32.71772526664741
  46.53715808725271
  63.50293439985202
  84.07304480374764
 108.64146078683832
 137.4725836395958
   ⋮
 756.3080584487884
 761.1195323523455
 765.2381316089154
 768.7669969454265
 771.7933477006857
 774.3892550109165
 776.6156094975315
 778.5257242512041
 780.1656522179535
````





The new cases per day are calculated from the cumulative counts.

````julia
X = C[2:end] .- C[1:(end-1)];
````


````
40-element Array{Float64,1}:
  5.5906270179533575
  7.076170331041077
  8.90898222880263
 11.141945688850349
 13.8194328206053
 16.96577631259931
 20.57011040389561
 24.568415983090688
 28.831122852757474
 33.14329091787582
  ⋮
  5.623429741450877
  4.811473903557044
  4.118599256569951
  3.528865336511103
  3.026350755259159
  2.595907310230814
  2.226354486615037
  1.9101147536725875
  1.639927966749383
````





Although the ODE system is deterministic, we can add measurement error to the counts of new cases. Here, a Poisson distribution is used, although a negative binomial could also be used (which would introduce an additional parameter for the variance).

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



````julia
Y = rand.(Poisson.(X));
````


````
40-element Array{Int64,1}:
  6
  9
  9
 11
 17
 21
 17
 22
 25
 34
  ⋮
  5
  3
  0
  1
  3
  3
  1
  2
  3
````





## Using Optim.jl directly

````julia
using Optim
````





### Single parameter optimization

This function calculates the sum of squares for a single parameter fit (β). Note how the original `ODEProblem` is remade using the `remake` function. Like all the loss functions listed here, `Inf` is returned if the number of daily cases is less than or equal to zero.

````julia
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
````


````
ss1 (generic function with 1 method)
````





Optimisation routines typically *minimise* functions, so for maximum likelihood estimates, we have to define the *negative* log-likelihood - here, for a single parameter, β.

````julia
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
````


````
nll1 (generic function with 1 method)
````





In this model, β is positive and (through the meaning of the parameter) bounded between 0 and 1. For point estimates, we could use constrained optimisation, or transform β to an unconstrained scale. Here is the first approach, defining the bounds and initial values for optimization.

````julia
lower1 = 0.0
upper1 = 1.0
initial_x1 = 0.1;
````


````
0.1
````





Model fit using sum of squares. The output isn't suppressed, as the output of the outcome of the optimisation, such as whether it has converged, is important.

````julia
opt1_ss = Optim.optimize(ss1,lower1,upper1)
````


````
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [0.000000, 1.000000]
 * Minimizer: 4.909134e-02
 * Minimum: 1.016203e+03
 * Iterations: 22
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16
): true
 * Objective Function Calls: 23
````





Model fit using (negative) log likelihood.

````julia
opt1_nll = Optim.optimize(nll1,lower1,upper1)
````


````
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [0.000000, 1.000000]
 * Minimizer: 4.969331e-02
 * Minimum: 1.117286e+02
 * Iterations: 24
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16
): true
 * Objective Function Calls: 25
````





### Multiparameter optimization

Multiple parameters are handled in the cost function using an array argument. Firstly, sum of squares.

````julia
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
````


````
ss2 (generic function with 1 method)
````





Secondly, negative log-likelihood.

````julia
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
````


````
nll2 (generic function with 1 method)
````





Two-parameter lower and upper bounds and initial conditions.

````julia
lower2 = [0.0,0.0]
upper2 = [1.0,1.0]
initial_x2 = [0.01,0.1];
````


````
2-element Array{Float64,1}:
 0.01
 0.1
````



````julia
opt2_ss = Optim.optimize(ss2,lower2,upper2,initial_x2)
````


````
* Status: success

 * Candidate solution
    Minimizer: [9.63e-03, 4.93e-02]
    Minimum:   1.013835e+03

 * Found with
    Algorithm:     Fminbox with L-BFGS
    Initial Point: [1.00e-02, 1.00e-01]

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 1.49e+00 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    3
    f(x) calls:    139
    ∇f(x) calls:   139
````



````julia
opt2_nll = Optim.optimize(nll2,lower2,upper2,initial_x2)
````


````
* Status: success

 * Candidate solution
    Minimizer: [1.00e-02, 4.97e-02]
    Minimum:   1.117286e+02

 * Found with
    Algorithm:     Fminbox with L-BFGS
    Initial Point: [1.00e-02, 1.00e-01]

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 5.76e-02 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    3
    f(x) calls:    165
    ∇f(x) calls:   165
````





## Using DiffEqParamEstim

The advantage of using a framework such as DiffEqParamEstim is that a number of different frameworks can be employed easily. Firstly, the loss function is defined.

````julia
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
````


````
loss_function (generic function with 1 method)
````





Secondly, a function that generates the `Problem` to be solved.

````julia
prob_generator = (prob,q) -> remake(prob,
    u0=[1000.0-(q[1]*1000),q[1]*1000,0.0,0.0],
    p=[q[2],10.0,0.25]);
````


````
#1 (generic function with 1 method)
````





The loss function and the problem generator then get combined to build the objective function.

````julia
cost_function = build_loss_objective(prob_ode,
    Tsit5(),
    loss_function,
    saveat=δt,
    prob_generator = prob_generator,
    maxiters=100,
    verbose=false);
````


````
(::DiffEqParamEstim.DiffEqObjective{DiffEqParamEstim.var"#43#48"{Nothing,Bo
ol,Int64,Main.##WeaveSandBox#785.var"#1#2",Base.Iterators.Pairs{Symbol,Real
,Tuple{Symbol,Symbol,Symbol},NamedTuple{(:saveat, :maxiters, :verbose),Tupl
e{Float64,Int64,Bool}}},DiffEqBase.ODEProblem{Array{Float64,1},Tuple{Float6
4,Float64},true,Array{Float64,1},DiffEqBase.ODEFunction{true,typeof(Main.##
WeaveSandBox#785.sir_ode!),LinearAlgebra.UniformScaling{Bool},Nothing,Nothi
ng,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,
Nothing},Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}
}},DiffEqBase.StandardODEProblem},OrdinaryDiffEq.Tsit5,typeof(Main.##WeaveS
andBox#785.loss_function),Nothing},DiffEqParamEstim.var"#47#53"{DiffEqParam
Estim.var"#43#48"{Nothing,Bool,Int64,Main.##WeaveSandBox#785.var"#1#2",Base
.Iterators.Pairs{Symbol,Real,Tuple{Symbol,Symbol,Symbol},NamedTuple{(:savea
t, :maxiters, :verbose),Tuple{Float64,Int64,Bool}}},DiffEqBase.ODEProblem{A
rray{Float64,1},Tuple{Float64,Float64},true,Array{Float64,1},DiffEqBase.ODE
Function{true,typeof(Main.##WeaveSandBox#785.sir_ode!),LinearAlgebra.Unifor
mScaling{Bool},Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Noth
ing,Nothing,Nothing,Nothing,Nothing},Base.Iterators.Pairs{Union{},Union{},T
uple{},NamedTuple{(),Tuple{}}},DiffEqBase.StandardODEProblem},OrdinaryDiffE
q.Tsit5,typeof(Main.##WeaveSandBox#785.loss_function),Nothing}}}) (generic 
function with 2 methods)
````





### Optim interface

The resulting cost function can be passed to `Optim.jl` as before.

````julia
opt_pe1 = Optim.optimize(cost_function,lower2,upper2,initial_x2)
````


````
* Status: success

 * Candidate solution
    Minimizer: [1.00e-02, 4.97e-02]
    Minimum:   1.117286e+02

 * Found with
    Algorithm:     Fminbox with L-BFGS
    Initial Point: [1.00e-02, 1.00e-01]

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 5.76e-02 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    3
    f(x) calls:    165
    ∇f(x) calls:   165
````





### NLopt interface

The same function can also be passed to `NLopt.jl`. For some reason, this reaches the maximum number of evaluations.

````julia
using NLopt
opt = Opt(:LD_MMA, 2)
opt.lower_bounds = lower2
opt.upper_bounds = upper2
opt.min_objective = cost_function
opt.maxeval = 10000
(minf,minx,ret) = NLopt.optimize(opt,initial_x2)
````


````
(111.72855334375784, [0.010000860197548981, 0.04969282386672637], :MAXEVAL_
REACHED)
````





### BlackBoxOptim interface

We can also use `BlackBoxOptim.jl`.

````julia
using BlackBoxOptim
bound1 = Tuple{Float64, Float64}[(0.0,1.0),(0.0, 1.0)]
result = bboptimize(cost_function;SearchRange = bound1, MaxSteps = 1e4)
````


````
Starting optimization with optimizer BlackBoxOptim.DiffEvoOpt{BlackBoxOptim
.FitPopulation{Float64},BlackBoxOptim.RadiusLimitedSelector,BlackBoxOptim.A
daptiveDiffEvoRandBin{3},BlackBoxOptim.RandomBound{BlackBoxOptim.Continuous
RectSearchSpace}}
0.00 secs, 0 evals, 0 steps

Optimization stopped after 10001 steps and 0.35 seconds
Termination reason: Max number of steps (10000) reached
Steps per second = 28402.16
Function evals per second = 28620.83
Improvements/step = 0.13950
Total function evaluations = 10078


Best candidate found: [0.0100005, 0.049693]

Fitness: 111.728553108

BlackBoxOptim.OptimizationResults("adaptive_de_rand_1_bin_radiuslimited", "
Max number of steps (10000) reached", 10001, 1.594788198760762e9, 0.3521211
1473083496, BlackBoxOptim.DictChain{Symbol,Any}[BlackBoxOptim.DictChain{Sym
bol,Any}[Dict{Symbol,Any}(:RngSeed => 446022,:SearchRange => [(0.0, 1.0), (
0.0, 1.0)],:MaxSteps => 10000),Dict{Symbol,Any}()],Dict{Symbol,Any}(:Fitnes
sScheme => BlackBoxOptim.ScalarFitnessScheme{true}(),:NumDimensions => :Not
Specified,:PopulationSize => 50,:MaxTime => 0.0,:SearchRange => (-1.0, 1.0)
,:Method => :adaptive_de_rand_1_bin_radiuslimited,:MaxNumStepsWithoutFuncEv
als => 100,:RngSeed => 1234,:MaxFuncEvals => 0,:SaveTrace => false…)], 1007
8, BlackBoxOptim.ScalarFitnessScheme{true}(), BlackBoxOptim.TopListArchiveO
utput{Float64,Array{Float64,1}}(111.72855310838251, [0.01000046836278904, 0
.049692985596648996]), BlackBoxOptim.PopulationOptimizerOutput{BlackBoxOpti
m.FitPopulation{Float64}}(BlackBoxOptim.FitPopulation{Float64}([0.010000441
003697182 0.010000196839751072 … 0.010000316998278278 0.010000255271509311;
 0.04969310950094767 0.049693077283398515 … 0.049693210077759314 0.04969306
084709772], NaN, [111.7285532227244, 111.72855315334847, 111.72855319568794
, 111.72855321074974, 111.72855324392246, 111.72855321051965, 111.728553198
35722, 111.72855324392246, 111.72855327721915, 111.72855310838251  …  111.7
2855318385034, 111.72855318505471, 111.72855322038515, 111.72855312105112, 
111.7285531784525, 111.72855316781599, 111.72855325364509, 111.728553194474
39, 111.72855324979908, 111.72855320326772], 0, BlackBoxOptim.Candidate{Flo
at64}[BlackBoxOptim.Candidate{Float64}([0.010000694535302126, 0.04969294508
41662], 18, 111.72855315905642, BlackBoxOptim.AdaptiveDiffEvoRandBin{3}(Bla
ckBoxOptim.AdaptiveDiffEvoParameters(BlackBoxOptim.BimodalCauchy(Distributi
ons.Cauchy{Float64}(μ=0.65, σ=0.1), Distributions.Cauchy{Float64}(μ=1.0, σ=
0.1), 0.5, false, true), BlackBoxOptim.BimodalCauchy(Distributions.Cauchy{F
loat64}(μ=0.1, σ=0.1), Distributions.Cauchy{Float64}(μ=0.95, σ=0.1), 0.5, f
alse, true), [0.5986560490312182, 0.8801497880474216, 1.0, 0.28784804165558
34, 0.9552552490374797, 1.0, 1.0, 0.8549312033008517, 0.23003692778296714, 
1.0  …  0.7284955649698605, 0.7073267424836166, 0.9416356618465631, 1.0, 0.
7680563512395949, 1.0, 0.6726221965612929, 0.8533619770082745, 0.6531367213
592068, 0.5826669890378492], [0.5529302294793459, 0.055586527848320175, 0.7
205786666309878, 0.7906824278551479, 0.7937381275976991, 0.6573477673499002
, 0.06965252304851306, 1.0, 1.0, 0.22843084603538916  …  0.9260678525364435
, 0.18802543518817297, 0.08171423015182257, 0.12190189158528267, 0.64512560
16364262, 0.9417241944379853, 0.005300130416607957, 0.16214417417590216, 0.
021737038767065342, 0.3092563744286215])), 0), BlackBoxOptim.Candidate{Floa
t64}([0.010000276199212788, 0.04969312984053147], 18, 111.72855317892403, B
lackBoxOptim.AdaptiveDiffEvoRandBin{3}(BlackBoxOptim.AdaptiveDiffEvoParamet
ers(BlackBoxOptim.BimodalCauchy(Distributions.Cauchy{Float64}(μ=0.65, σ=0.1
), Distributions.Cauchy{Float64}(μ=1.0, σ=0.1), 0.5, false, true), BlackBox
Optim.BimodalCauchy(Distributions.Cauchy{Float64}(μ=0.1, σ=0.1), Distributi
ons.Cauchy{Float64}(μ=0.95, σ=0.1), 0.5, false, true), [0.5986560490312182,
 0.8801497880474216, 1.0, 0.2878480416555834, 0.9552552490374797, 1.0, 1.0,
 0.8549312033008517, 0.23003692778296714, 1.0  …  0.7284955649698605, 0.707
3267424836166, 0.9416356618465631, 1.0, 0.7680563512395949, 1.0, 0.67262219
65612929, 0.8533619770082745, 0.6531367213592068, 0.5826669890378492], [0.5
529302294793459, 0.055586527848320175, 0.7205786666309878, 0.79068242785514
79, 0.7937381275976991, 0.6573477673499002, 0.06965252304851306, 1.0, 1.0, 
0.22843084603538916  …  0.9260678525364435, 0.18802543518817297, 0.08171423
015182257, 0.12190189158528267, 0.6451256016364262, 0.9417241944379853, 0.0
05300130416607957, 0.16214417417590216, 0.021737038767065342, 0.30925637442
86215])), 0)])))
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
[80f14c24-f653-4e6a-9b94-39d6b0f70001] AbstractMCMC 1.0.1
[537997a7-5e4e-5d89-9595-2241ea00577e] AbstractPlotting 0.12.3
[46ada45e-f475-11e8-01d0-f70cc89e6671] Agents 3.2.1
[4f99eebe-17bf-4e98-b6a1-2c4f205a959b] AlgebraicPetri 0.3.1
[f5f396d3-230c-5e07-80e6-9fadf06146cc] ApproxBayes 0.3.2
[c52e3926-4ff0-5f6e-af25-54175e0327b1] Atom 0.12.16
[fbb218c0-5317-5bc6-957e-2ee96dd4b1f0] BSON 0.2.6
[6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf] BenchmarkTools 0.5.0
[a134a8b2-14d6-55f6-9291-3336d3ab0209] BlackBoxOptim 0.5.0
[2d3116d5-4b8f-5680-861c-71f149790274] Bridge 0.11.3
[1aa9af3a-2424-508f-bb7e-0626de155470] BridgeDiffEq 0.1.0
[46d747a0-b9e1-11e9-14b5-615c73e45078] BridgeSDEInference 0.3.2
[336ed68f-0bac-5ca0-87d4-7b16caf5d00b] CSV 0.7.3
[49dc2e85-a5d0-5ad3-a950-438e2897f1b9] Calculus 0.5.1
[134e5e36-593f-5add-ad60-77f754baafbe] Catlab 0.7.1
[aaaa29a8-35af-508c-8bc3-b662a17a0fe5] Clustering 0.14.1
[2445eb08-9709-466a-b3fc-47e12bd697a2] DataDrivenDiffEq 0.3.1
[a93c6f00-e57d-5684-b7b6-d8193f3e46c0] DataFrames 0.21.4
[7806a523-6efd-50cb-b5f6-3fa6f1930dbb] DecisionTree 0.10.6
[bcd4f6db-9728-5f36-b5f7-82caef46ccdb] DelayDiffEq 5.24.1
[2b5f629d-d688-5b77-993f-72d75c75574e] DiffEqBase 6.40.7
[ebbdde9d-f333-5424-9be2-dbf1e9acfb5e] DiffEqBayes 2.16.0
[eb300fae-53e8-50a0-950c-e21f52c2b7e0] DiffEqBiological 4.3.0
[459566f4-90b8-5000-8ac3-15dfb0a30def] DiffEqCallbacks 2.13.3
[aae7a2af-3d4f-5e19-a356-7da93b79d9d0] DiffEqFlux 1.17.0
[c894b116-72e5-5b58-be3c-e6d8d4ac2b12] DiffEqJump 6.9.3
[1130ab10-4a5a-5621-a13d-e4788d82bd4c] DiffEqParamEstim 1.16.0
[41bf760c-e81c-5289-8e54-58b1f1f8abe2] DiffEqSensitivity 6.23.0
[0c46a032-eb83-5123-abaf-570d42b7fbaa] DifferentialEquations 6.15.0
[b4f34e82-e78d-54a5-968a-f98e89d6e8f7] Distances 0.9.0
[31c24e10-a181-5473-b8eb-7969acd0382f] Distributions 0.23.4
[634d3b9d-ee7a-5ddf-bec9-22491ea816e1] DrWatson 1.14.4
[f6006082-12f8-11e9-0c9c-0d5d367ab1e5] EvoTrees 0.4.9
[587475ba-b771-5e3f-ad9e-33799f191a9c] Flux 0.10.4
[f6369f11-7733-5829-9624-2563aa707210] ForwardDiff 0.10.12
[38e38edf-8417-5370-95a0-9cbb8c7f171a] GLM 1.3.9
[28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71] GR 0.50.1
[891a1506-143c-57d2-908e-e1f8e92e6de9] GaussianProcesses 0.12.1
[ea4f424c-a589-11e8-07c0-fd5c91b9da4a] Gen 0.3.5
[523d8e89-b243-5607-941c-87d699ea6713] Gillespie 0.1.0
[e850a1a4-d859-11e8-3d54-a195e6d045d3] GpABC 0.1.1
[7073ff75-c697-5162-941a-fcdaad2a7d2a] IJulia 1.21.2
[a98d9a8b-a2ab-59e6-89dd-64a1c18fca59] Interpolations 0.12.10
[c8e1da08-722c-5040-9ed9-7db0dc04731e] IterTools 1.3.0
[4076af6c-e467-56ae-b986-b466b2749572] JuMP 0.21.3
[e5e0dc1b-0480-54bc-9374-aad01c23163d] Juno 0.8.2
[b1bec4e5-fd48-53fe-b0cb-9723c09d164b] LIBSVM 0.4.0
[b964fa9f-0449-5b57-a5c2-d3ea65f4040f] LaTeXStrings 1.1.0
[2ee39098-c373-598a-b85f-a56591580800] LabelledArrays 1.3.0
[23fbe1c1-3f47-55db-b15f-69d7ec21a316] Latexify 0.13.5
[7acf609c-83a4-11e9-1ffb-b912bcd3b04a] LightGBM 0.3.1
[093fc24a-ae57-5d10-9952-331d41423f4d] LightGraphs 1.3.3
[30fc2ffe-d236-52d8-8643-a9d8f7c094a7] LossFunctions 0.6.2
[c7f686f2-ff18-58e9-bc7b-31028e88f75d] MCMCChains 4.0.1
[add582a8-e3ab-11e8-2d5e-e98b27df1bc7] MLJ 0.12.0
[094fc8d1-fd35-5302-93ea-dabda2abf845] MLJFlux 0.1.2
[6ee0df7b-362f-4a72-a706-9e79364fb692] MLJLinearModels 0.5.0
[d491faf4-2d78-11e9-2867-c94bc002c0b7] MLJModels 0.11.0
[1914dd2f-81c6-5fcd-8719-6d5c9610ff09] MacroTools 0.5.5
[5424a776-8be3-5c5b-a13f-3551f69ba0e6] Mamba 0.12.4
[ff71e718-51f3-5ec2-a782-8ffcbfa3c316] MixedModels 3.0.0-DEV
[961ee093-0014-501f-94e3-6117800e7a78] ModelingToolkit 3.13.0
[6f286f6a-111f-5878-ab1e-185364afe411] MultivariateStats 0.7.0
[76087f3c-5699-56af-9a33-bf431cd00edd] NLopt 0.6.0
[9bbee03b-0db5-5f46-924f-b5c9c21b8c60] NaiveBayes 0.4.0
[b8a86587-4115-5ab1-83bc-aa920d37bbce] NearestNeighbors 0.4.6
[41ceaf6f-1696-4a54-9b49-2e7a9ec3782e] NestedSamplers 0.4.0
[47be7bcc-f1a6-5447-8b36-7eeeff7534fd] ORCA 0.4.0
[429524aa-4258-5aef-a3af-852621145aeb] Optim 0.21.0
[1dea7af3-3e70-54e6-95c3-0bf5283fa5ed] OrdinaryDiffEq 5.41.0
[42b8e9d4-006b-409a-8472-7f34b3fb58af] ParallelKMeans 0.1.8
[4259d249-1051-49fa-8328-3f8ab9391c33] Petri 1.1.0
[91a5bcdd-55d7-5caf-9e0b-520d859cae80] Plots 1.5.4
[c3e4b0f8-55cb-11ea-2926-15256bba5781] Pluto 0.10.6
[d330b81b-6aea-500a-939a-2ce795aea3ee] PyPlot 2.9.0
[1a8c2f83-1ff3-5112-b086-8aa67b057ba1] Query 0.12.3-DEV
[6f49c342-dc21-5d91-9882-a32aef131414] RCall 0.13.7
[e6cf234a-135c-5ec9-84dd-332b85af5143] RandomNumbers 1.4.0
[c5292f4c-5179-55e1-98c5-05642aab7184] ResumableFunctions 0.5.1
[37e2e3b7-166d-5795-8a7a-e32c996b4267] ReverseDiff 1.2.0
[3646fa90-6ef7-5e7e-9f22-8aca16db6324] ScikitLearn 0.6.2
[f5ac2a72-33c7-5caf-b863-f02fefdcf428] SemanticModels 0.3.0
[428bdadb-6287-5aa5-874b-9969638295fd] SimJulia 0.8.0
[05bca326-078c-5bf0-a5bf-ce7c7982d7fd] SimpleDiffEq 1.1.0
[276daf66-3868-5448-9aa4-cd146d93841b] SpecialFunctions 0.10.3
[5a560754-308a-11ea-3701-ef72685e98d6] Splines2 0.1.0
[2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91] StatsBase 0.33.0
[f3b207a7-027a-5e70-b257-86293d7955fd] StatsPlots 0.14.6
[789caeaf-c7a9-5a7d-9973-96adeb23e2a0] StochasticDiffEq 6.24.0
[92b13dbe-c966-51a2-8445-caca9f8a7d42] TaylorIntegration 0.8.3
[9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c] Tracker 0.2.8
[fce5fe82-541a-59a6-adf8-730c64b5f9a0] Turing 0.13.0
[1986cc42-f94f-5a68-af5c-568840ba703d] Unitful 1.3.0
[276b4fcb-3e11-5398-bf8b-a0c2d153d008] WGLMakie 0.2.5
[29a6e085-ba6d-5f35-a997-948ac2efa89a] Wavelets 0.9.2
[44d3d7a6-8a23-5bf8-98c5-b353f8df5ec9] Weave 0.10.2
[009559a3-9522-5dbb-924b-0b6ed2b22bb9] XGBoost 1.1.1
```
