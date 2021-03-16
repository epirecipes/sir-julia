
using DifferentialEquations
using OrdinaryDiffEq
using MomentClosure
using ModelingToolkit
using DiffEqJump
using DataFrames
using Statistics
using Plots
using StatsPlots
using BenchmarkTools


@parameters t β c γ
@variables S(t) I(t);


rxs = [Reaction(β*c, [S,I], [I], [1,1], [2])
       Reaction(γ, [I], nothing)]
rs  = ReactionSystem(rxs, t, [S,I], [β,c,γ]);


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
ts = 0:δt:tmax;


u0 = [S => 990.0, I => 10.0];


u0v = [x[2] for x in u0];


p = [β=>0.00005, c=>10.0, γ=>0.25];


central_eqs = generate_central_moment_eqs(rs, 4, combinatoric_ratelaw=false);


closure_methods = ["zero","normal","log-normal","gamma","derivative matching"];


closed_central_eqs = Dict(cm=>moment_closure(central_eqs,cm) for cm in closure_methods);


u0map = Dict(cm=> deterministic_IC(u0v,closed_central_eqs[cm]) for cm in closure_methods);


closed_central_eqs_df = Dict{String,DataFrame}()
for cm in closure_methods
    prob = ODEProblem(closed_central_eqs[cm], u0map[cm], tspan, p)
    sol = solve(prob)
    df = DataFrame(sol(ts)',[replace(string(x[1]),"(t)" => "") for x in u0map[cm]])
    df[!,:t] = ts
    closed_central_eqs_df[cm] = df
end;


jumpsys = convert(JumpSystem, rs)
u0i = [S => 990, I => 10]
dprob = DiscreteProblem(jumpsys, u0i, tspan, p)
jprob = JumpProblem(jumpsys, dprob, Direct());


ensemble_jprob = EnsembleProblem(jprob)
ensemble_jsol = solve(ensemble_jprob,SSAStepper(),trajectories=20000)
ensemble_summary = EnsembleSummary(ensemble_jsol,ts);


ensemble_u = DataFrame(ensemble_summary.u',["μ₁₀","μ₀₁"])
ensemble_v = DataFrame(ensemble_summary.v',["M₂₀","M₀₂"])
ensemble_uv = hcat(ensemble_u,ensemble_v)
ensemble_uv[!,:t] = ts;


jplot = @df ensemble_uv plot(:t,[:μ₁₀,:μ₀₁],
     ribbon=[2*sqrt.(:M₂₀),
             2*sqrt.(:M₀₂)],
     label=["S" "I"],
     xlabel="Time",
     ylabel="Number",
     title="Jump process")


pltlist = []
for cm in closure_methods
     plt = @df closed_central_eqs_df[cm] plot(:t,[:μ₁₀,:μ₀₁],
          ribbon=[2*sqrt.(:M₂₀),
                  2*sqrt.(:M₀₂)],
          label=["S" "I"],
          xlabel="Time",
          ylabel="Number",
          title=cm)
    push!(pltlist,plt)
end;


l = @layout [a b c; d e f]
plot(vcat(jplot,pltlist)...,layout=l)


m = []
c = []
v = []
for moment in [:μ₁₀,:μ₀₁,:M₂₀,:M₀₂]
    for cm in closure_methods
        push!(m,moment)
        push!(c,cm)
        push!(v,mean(abs.(closed_central_eqs_df[cm][!,moment] - ensemble_uv[!,moment])))
    end
end
print(DataFrame(["Moment" => m,"Method" => c,"Normalized L1" => v]));


@benchmark solve(ensemble_jprob,SSAStepper(),trajectories=20000)


prob = ODEProblem(closed_central_eqs["normal"], u0map["normal"], tspan, p)
@benchmark sol = solve(prob)

