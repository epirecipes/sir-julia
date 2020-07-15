
using AlgebraicPetri.Epidemiology
using Petri
using Catlab.Theories
using Catlab.CategoricalAlgebra.ShapeDiagrams
using Catlab.Graphics
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqJump
using Random
using Plots

# helper function to visualize categorical representation
display_wd(ex) = to_graphviz(ex, orientation=LeftToRight, labels=true);


@present InfectiousDiseases(FreeBiproductCategory) begin
    S::Ob
    E::Ob
    I::Ob
    R::Ob
    D::Ob
    transmission::Hom(S⊗I, I)
    exposure::Hom(S⊗I, E⊗I)
    illness::Hom(E,I)
    recovery::Hom(I,R)
    death::Hom(I,D)
end


ob = PetriCospanOb(1)
spontaneous_petri = PetriCospan([1], Petri.Model(1:2, [(Dict(1=>1), Dict(2=>1))]), [2])
transmission_petri = PetriCospan([1], Petri.Model(1:2, [(Dict(1=>1, 2=>1), Dict(2=>2))]), [2])
exposure_petri = PetriCospan([1, 2], Petri.Model(1:3, [(Dict(1=>1, 2=>1), Dict(3=>1, 2=>1))]), [3, 2])

const FunctorGenerators = Dict(S=>ob, E=>ob, I=>ob, R=>ob, D=>ob,
        transmission=>transmission_petri, exposure=>exposure_petri,
        illness=>spontaneous_petri, recovery=>spontaneous_petri, death=>spontaneous_petri)


Graph(decoration(F_epi(transmission)))


Graph(decoration(F_epi(recovery)))


sir_wiring_diagram = transmission ⋅ recovery
display_wd(sir_wiring_diagram)


sir_model = decoration(F_epi(sir_wiring_diagram));
Graph(sir_model)


tmax = 40.0
tspan = (0.0,tmax);


u0 = [990,10,0]; # S,I,R


p = [0.05*10.0/sum(u0),0.25]; # β*c/N,γ


Random.seed!(1234);


prob_ode = ODEProblem(sir_model,u0,tspan,p)
sol_ode = solve(prob_ode, Tsit5());
plot(sol_ode)


prob_sde,cb = SDEProblem(sir_model,u0,tspan,p)
sol_sde = solve(prob_sde,LambaEM(),callback=cb);
plot(sol_sde)


prob_jump = JumpProblem(sir_model, u0, tspan, p)
sol_jump = solve(prob_jump,SSAStepper());
plot(sol_jump)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

