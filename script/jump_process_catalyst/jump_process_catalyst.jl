
using Catalyst
using DiffEqJump
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


sir_model = @reaction_network begin
  β*c/(s+i+r), s + i --> 2i
  γ, i --> r
end β c γ;


tmax = 40.0
tspan = (0.0,tmax);


δt = 0.1
t = 0:δt:tmax;


u0 = [990,10,0]; # S,I,R


p = [0.05,10.0,0.25];


Random.seed!(1234);


prob_discrete = DiscreteProblem(sir_model,u0,tspan,p);


prob_jump = JumpProblem(sir_model,prob_discrete,Direct());


sol_jump = solve(prob_jump,SSAStepper());


out_jump = sol_jump(t);


df_jump = DataFrame(out_jump')
df_jump[!,:t] = out_jump.t;


@df df_jump plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_jump,SSAStepper())

