
using DifferentialEquations
using SimpleDiffEq
using Distributions
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


@inline function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end


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
end


δt = 0.1
nsteps = 400
tmax = nsteps*δt
tspan = (0.0,nsteps)
t = 0.0:δt:tmax;


u0 = [990,10,0]


p = [0.05,10.0,0.25,δt]


Random.seed!(1234);


prob_markov = DiscreteProblem(sir_markov!,u0,tspan,p)


sol_markov = solve(prob_markov,solver=FunctionMap);


df_markov = DataFrame(sol_markov')
df_markov[!,:t] = t;


@df df_markov plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_markov,solver=FunctionMap)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

