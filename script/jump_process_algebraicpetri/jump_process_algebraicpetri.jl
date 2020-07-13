
using AlgebraicPetri.Epidemiology
using Petri
using Catlab.Theories
using Catlab.CategoricalAlgebra.ShapeDiagrams
using Catlab.Graphics
using DiffEqJump
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


tmax = 40.0
tspan = (0.0,tmax);


δt = 0.1
t = 0:δt:tmax;


u0 = [990,10,0]; # S,I,R


p = [0.05*10.0/sum(u0),0.25]; # β*c/N,γ


Random.seed!(1234);


prob_jump = JumpProblem(sir_model, u0, tspan, p)


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


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

