
using DifferentialEquations
using SimpleDiffEq
using Distributions
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


function sir_sde!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ,δt) = p
    N = S+I+R
    ifrac = β*c*I/N*S*δt
    rfrac = γ*I*δt
    ifrac_noise = sqrt(ifrac)*rand(Normal(0,1))
    rfrac_noise = sqrt(rfrac)*rand(Normal(0,1))
    @inbounds begin
        du[1] = S-(ifrac+ifrac_noise)
        du[2] = I+(ifrac+ifrac_noise) - (rfrac + rfrac_noise)
        du[3] = R+(rfrac+rfrac_noise)
    end
    for i in 1:3
        if du[i] < 0 du[i]=0 end
    end
    nothing
end;


δt = 0.1
nsteps = 400
tmax = nsteps*δt
tspan = (0.0,nsteps)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]; # S,I,R


p = [0.05,10.0,0.25,δt]; # β,c,γ,δt


Random.seed!(1234);


prob_sde = DiscreteProblem(sir_sde!,u0,tspan,p)


sol_sde = solve(prob_sde,solver=FunctionMap);


df_sde = DataFrame(sol_sde')
df_sde[!,:t] = t;


@df df_sde plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_sde,solver=FunctionMap)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

