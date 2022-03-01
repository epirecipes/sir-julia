
using DifferentialEquations
using SimpleDiffEq
using Tables
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


u0 = [990.0,10.0,0.0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem(sir_ode!, u0, tspan, p);


sol_ode = solve(prob_ode, dt = δt);


df_ode = DataFrame(Tables.table(sol_ode'))
rename!(df_ode,["S","I","R"])
df_ode[!,:t] = sol_ode.t;


@df df_ode plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_ode, dt = δt);

