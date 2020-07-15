
using Petri
using LabelledArrays
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqJump
using Random
using Plots


sir = Petri.Model([:S,:I,:R],LVector(
                                inf=(LVector(S=1,I=1), LVector(I=2)),
                                rec=(LVector(I=1),     LVector(R=1))))


Graph(sir)


tmax = 40.0
tspan = (0.0,tmax);


u0 = LVector(S=990.0, I=10.0, R=0.0)


p = LVector(inf=0.5/sum(u0), rec=0.25);


Random.seed!(1234);


prob_ode = ODEProblem(sir,u0,tspan,p)
sol_ode = solve(prob_ode, Tsit5());
plot(sol_ode)


prob_sde,cb = SDEProblem(sir,u0,tspan,p)
sol_sde = solve(prob_sde,LambaEM(),callback=cb);
plot(sol_sde)


prob_jump = JumpProblem(sir, u0, tspan, p)
sol_jump = solve(prob_jump,SSAStepper());
plot(sol_jump)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

