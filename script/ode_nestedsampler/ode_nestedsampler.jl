
using DifferentialEquations
using SimpleDiffEq
using DiffEqSensitivity
using Random
using Distributions
using NestedSamplers
using StatsBase: sample, Weights
using MCMCChains: Chains, describe
using StatsPlots


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


δt = 1.0
tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax;


u0 = [990.0,10.0,0.0,0.0]; # S,I.R,Y


p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode,Tsit5(),saveat=δt);


out = Array(sol_ode)
C = out[4,:];


X = C[2:end] .- C[1:(end-1)];


Random.seed!(1234);


Y = rand.(Poisson.(X));


function ll(x)
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
    sum(logpdf.(Poisson.(X),Y))
end;


priors = [
    Uniform(0, 0.1),
    Uniform(0, 0.1)
]


model = NestedModel(ll, priors);


spl = Nested(2, 10000, bounds=Bounds.MultiEllipsoid)


chain = sample(model, spl;
               param_names=["i0", "β"],
               chain_type=Chains)


describe(chain)


plot(chain)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

