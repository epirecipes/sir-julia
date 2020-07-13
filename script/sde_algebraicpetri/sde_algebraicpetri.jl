
using AlgebraicPetri.Epidemiology
using Petri
using Catlab.Theories
using Catlab.CategoricalAlgebra.ShapeDiagrams
using Catlab.Graphics
using StochasticDiffEq
using Random
using DataFrames
using StatsPlots
using BenchmarkTools
import Base: ≤

# helper function to visualize categorical representation
display_wd(ex) = to_graphviz(ex, orientation=LeftToRight, labels=true);


Graph(decoration(F_epi(transmission)))


Graph(decoration(F_epi(recovery)))


sir_wiring_diagram = transmission ⋅ recovery
display_wd(sir_wiring_diagram)


sir_model = decoration(F_epi(sir_wiring_diagram));
Graph(sir_model)


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]; # S,I,R


p = [0.05*10.0/sum(u0),0.25]; # β*c/N,γ


Random.seed!(1234);


prob_sde,cb = SDEProblem(sir_model,u0,tspan,p)


sol_sde = solve(prob_sde,SRA1(),callback=cb);


df_sde = DataFrame(sol_sde(t)')
df_sde[!,:t] = t;


@df df_sde plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_sde,SRA1(),callback=cb)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

