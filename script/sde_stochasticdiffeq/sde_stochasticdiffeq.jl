
using DifferentialEquations
using StochasticDiffEq
using DiffEqCallbacks
using Random
using SparseArrays
using DataFrames
using StatsPlots
using BenchmarkTools


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


# Define a sparse matrix by making a dense matrix and setting some values as not zero
A = zeros(3,2)
A[1,1] = 1
A[2,1] = 1
A[2,2] = 1
A[3,2] = 1
A = SparseArrays.sparse(A);


# Make `g` write the sparse matrix values
function sir_noise!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    ifrac = β*c*I/N*S
    rfrac = γ*I
    du[1,1] = -sqrt(ifrac)
    du[2,1] = sqrt(ifrac)
    du[2,2] = -sqrt(rfrac)
    du[3,2] = sqrt(rfrac)
end;


function condition(u,t,integrator) # Event when event_f(u,t) == 0
  u[2]
end


function affect!(integrator)
  integrator.u[2] = 0.0
end


cb = ContinuousCallback(condition,affect!)


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


prob_sde = SDEProblem(sir_ode!,sir_noise!,u0,tspan,p,noise_rate_prototype=A)


sol_sde = solve(prob_sde,LambaEM(),callback=cb);


df_sde = DataFrame(sol_sde(t)')
df_sde[!,:t] = t;


@df df_sde plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_sde,LambaEM(),callback=cb)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

