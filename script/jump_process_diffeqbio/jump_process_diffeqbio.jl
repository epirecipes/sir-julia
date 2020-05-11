
using DiffEqBiological
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


sir_model = @reaction_network sir_rn begin
  0.5/1000, s + i --> 2i
  0.25, i --> r
end


tmax = 40.0
tspan = (0.0,tmax);


δt = 0.1
t = 0:δt:tmax;


u0 = [990,10,0]; # S,I,R


Random.seed!(1234);


prob = DiscreteProblem(u0,tspan)


prob_jump = JumpProblem(prob,Direct(),sir_model)


sol_jump = solve(prob_jump,SSAStepper());


out_jump = sol_jump(t);


df_jump = DataFrame(out_jump')
df_jump[!,:t] = out_jump.t;


@df df_jump plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_jump,FunctionMap())


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

