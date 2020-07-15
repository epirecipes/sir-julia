
using DifferentialEquations
using ModelingToolkit
using OrdinaryDiffEq
using DataFrames
using StatsPlots
using BenchmarkTools


@parameters t β c γ
@variables S(t) I(t) R(t)
@derivatives D'~t
N=S+I+R # This is recognized as a derived variable
eqs = [D(S) ~ -β*c*I/N*S,
       D(I) ~ β*c*I/N*S-γ*I,
       D(R) ~ γ*I];


sys = ODESystem(eqs)


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [S => 990.0,
      I => 10.0,
      R => 0.0];


p = [β=>0.05,
     c=>10.0,
     γ=>0.25];


prob_ode = ODEProblem(sys,u0,tspan,p;jac=true)


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

