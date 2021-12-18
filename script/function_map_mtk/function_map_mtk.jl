
using ModelingToolkit
using DifferentialEquations
using Distributions
using Tables
using DataFrames
using StatsPlots
using BenchmarkTools


@inline function rate_to_proportion(r,t)
    1-exp(-r*t)
end;


@parameters β=0.05 c=10.0 γ=0.25 N=1000.0 δt=0.1


@variables t SI(t)=0.0 IR(t)=0.0 S(t)=990.0 I(t)=10.0 R(t)=0.0


D = DiscreteUpdate(t; dt=δt)
eqs = [D(SI) ~ rate_to_proportion(β*c*I/N,δt)*S,
       D(IR) ~ rate_to_proportion(γ,δt)*I,
       D(S) ~ S-SI,
       D(I) ~ I+SI-IR,
       D(R) ~ R+IR]


@named sys = DiscreteSystem(eqs)


@named sys = DiscreteSystem(eqs, t, [SI, IR, S, I, R], [β,c,γ,N,δt])


prob = DiscreteProblem(sys, [], (0,400), [])


u0 = [S => 990.0,
      I => 10.0,
      R => 0.0];
p = [β=>0.05,
     c=>10.0,
     γ=>0.25,
     N=>1000.0,
     δt=>0.1];
prob = DiscreteProblem(sys, u0, (0,400), p)


sol = solve(prob,solver=FunctionMap);


df = DataFrame(Tables.table(sol'))
rename!(df,["SI", "IR", "S", "I", "R"])
df[!,:t] = 0:0.1:40.0;


@df df plot(:t,
    [:S :I :R],
    xlabel="Time",
    ylabel="Number")


C = cumsum(df[!,:SI])
cases = vcat(C[1:9],C[10:end] .- C[1:(end-9)])
df[!,"cases"] = cases


@df df plot(:t,
    [:cases],
    xlabel="Time",
    ylabel="Cases per day")


@benchmark solve(prob,solver=FunctionMap)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

