
using DifferentialEquations
using SimpleDiffEq
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


function infection_rate(u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    β*c*I/N*S
end
function infection!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
infection_jump = ConstantRateJump(infection_rate,infection!);


function recovery_rate(u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    γ*I
end
function recovery!(integrator)
  integrator.u[2] -= 1
  integrator.u[3] += 1
end
recovery_jump = ConstantRateJump(recovery_rate,recovery!);


tmax = 40.0
tspan = (0.0,tmax);


δt = 0.1
t = 0:δt:tmax;


u0 = [990,10,0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


prob = DiscreteProblem(u0,tspan,p)


prob_jump = JumpProblem(prob,Direct(),infection_jump,recovery_jump)


sol_jump = solve(prob_jump,FunctionMap());


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

