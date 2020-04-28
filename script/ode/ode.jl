
using DifferentialEquations
using SimpleDiffEq
using DataFrames
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


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]


p = [0.05,10.0,0.25];


prob_ode = ODEProblem(sir_ode!,u0,tspan,p)


sol_ode = solve(prob_ode);


df_ode = DataFrame(sol_ode(t)')
df_ode[!,:t] = t;


@df df_ode plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_ode)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

