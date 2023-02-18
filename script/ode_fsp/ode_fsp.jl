
using Catalyst
using ModelingToolkit
using FiniteStateProjection
using OrdinaryDiffEq
using JumpProcesses
using Random
using Plots
using StatsPlots


rn = @reaction_network SIR begin
    β, S + I --> 2I
    γ, I --> 0
end β γ


@parameters t β γ
@variables S(t) I(t)
rxs = [Reaction(β, [S,I], [I], [1,1], [2])
       Reaction(γ, [I], [])]
@named rs  = ReactionSystem(rxs, t, [S,I], [β,γ])


p = [0.005, 0.25]
u0 = [99, 1]
δt = 1.0
tspan = (0.0, 40.0)
solver = Tsit5()


prob_ode = ODEProblem(rs, u0, tspan, p)
sol_ode = solve(prob_ode, solver)
plot(sol_ode)


Random.seed!(1)
jumpsys = convert(JumpSystem, rs)
dprob = DiscreteProblem(jumpsys, u0, tspan, p)
jprob = JumpProblem(jumpsys, dprob, Direct())
jsol = solve(jprob, SSAStepper())
plot(jsol)


ensemble_jprob = EnsembleProblem(jprob)
ensemble_jsol = solve(ensemble_jprob,SSAStepper(),trajectories=10000);


jstates = [s(20) for s in ensemble_jsol]
histogram([s[1] for s in jstates], label="S", normalize=:pdf, alpha=0.5)
histogram!([s[2] for s in jstates], label="I", normalize=:pdf, alpha=0.5)


sys_fsp = FSPSystem(rn) # or FSPSystem(rs)
u0f = zeros(101, 101) # 2D system as we have two states
u0f[100,2] = 1.0 # this is equivalent to setting S(0)=99 and I(0)=1
prob_fsp = convert(ODEProblem, sys_fsp, u0f, tspan, p)
sol_fsp = solve(prob_fsp, solver, dense=false, saveat=δt);


bar(0:1:100,
    sum(sol_fsp.u[21],dims=2),
    xlabel="Number",
    ylabel="Probability",
    label="S",
    title="t="*string(sol_fsp.t[21]),
    alpha=0.5)
bar!(0:1:100,
     sum(sol_fsp.u[21],dims=1)',
     label="I",
     alpha=0.5)


p1 = bar(0:1:100, sum(sol_fsp.u[21],dims=2), label="S", title="t="*string(sol_fsp.t[21]), ylims=(0,1), alpha=0.5)
bar!(p1, 0:1:100, sum(sol_fsp.u[21],dims=1)', label="I",alpha=0.5)
p2 = bar(0:1:100, sum(sol_fsp.u[41],dims=2), label="S", title="t="*string(sol_fsp.t[41]), ylims=(0,1), alpha=0.5)
bar!(p2, 0:1:100, sum(sol_fsp.u[41],dims=1)', label="I", alpha=0.5)
l = @layout [a b]
plot(p1, p2, layout=l)

