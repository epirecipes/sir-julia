
using ModelingToolkit
using DiffEqJump
using Random
using DataFrames
using Tables
using StatsPlots
using BenchmarkTools


@parameters β c γ
@variables t S(t) I(t) R(t)
N = S+I+R
rate₁   = β*c*S*I/N
affect₁ = [S ~ S - 1, I ~ I + 1]
rate₂   = γ*I
affect₂ = [I ~ I - 1, R ~ R + 1]
j₁      = ConstantRateJump(rate₁,affect₁)
j₂      = ConstantRateJump(rate₂,affect₂)
@named sir_js = JumpSystem([j₁,j₂], t, [S,I,R], [β,c,γ])


tmax = 40.0
tspan = (0.0,tmax)
δt = 1.0;


u0 = [S => 990, I => 10, R => 0];


p = [β => 0.05, c => 10.0, γ => 0.25];


Random.seed!(1234);


sir_dprob = DiscreteProblem(sir_js, u0, tspan,p);


sir_jprob = JumpProblem(sir_js, sir_dprob,Direct());


sol_jump = solve(sir_jprob,SSAStepper());


out_jump = sol_jump(0:δt:tmax);


df_jump = DataFrame(Tables.table(out_jump'))
rename!(df_jump,["S","I","R"])
df_jump[!,:t] = out_jump.t;


@df df_jump plot(:t,
    [:S :I :R],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(sir_jprob,SSAStepper())

