
using DifferentialEquations
using ModelingToolkit
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqJump
using Random
using Plots


@parameters t β c γ
@variables S(t) I(t) R(t)

N=S+I+R # This is recognized as a derived variable
rxs = [Reaction((β*c)/N, [S,I], [I], [1,1], [2])
       Reaction(γ, [I], [R])]


rs  = ReactionSystem(rxs, t, [S,I,R], [β,c,γ])


tmax = 40.0
tspan = (0.0,tmax);


u0 = [S => 990.0,
      I => 10.0,
      R => 0.0];


p = [β=>0.05,
     c=>10.0,
     γ=>0.25];


Random.seed!(1234);


odesys = convert(ODESystem, rs)
oprob = ODEProblem(odesys, u0, tspan, p)
osol = solve(oprob, Tsit5())
plot(osol)


sdesys = convert(SDESystem, rs)
sprob = SDEProblem(sdesys, u0, tspan, p)
ssol = solve(sprob, LambaEM())
plot(ssol)


jumpsys = convert(JumpSystem, rs)
u0i = [S => 990, I => 10, R => 0]
dprob = DiscreteProblem(jumpsys, u0i, tspan, p)
jprob = JumpProblem(jumpsys, dprob, Direct())
jsol = solve(jprob, SSAStepper())
plot(jsol)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

