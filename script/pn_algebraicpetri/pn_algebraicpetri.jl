
using AlgebraicPetri
using AlgebraicPetri.Epidemiology

using Catlab
using Catlab.Graphics
using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra
using Catlab.Programs.RelationalPrograms

using LabelledArrays
using OrdinaryDiffEq
using Random
using Plots

# helper function to visualize categorical representation
display_uwd(ex) = to_graphviz(ex, box_labels=:name, junction_labels=:variable, edge_attrs=Dict(:len=>".75"));


# population x spontaneously moves to population y
spontaneous_petri(x::Symbol, y::Symbol, transition::Symbol) =
    Open(LabelledPetriNet(unique([x,y]), transition=>(x, y)))
# population y causes population x to move to population z
exposure_petri(x::Symbol, y::Symbol, z::Symbol, transition::Symbol) =
    Open(LabelledPetriNet(unique([x,y,z]), transition=>((x,y)=>(z,y))))

infection = exposure_petri(:S, :I, :I, :inf)
exposure = exposure_petri(:S, :I, :E, :exp)
illness = spontaneous_petri(:E,:I,:ill)
recovery = spontaneous_petri(:I,:R,:rec)
death = spontaneous_petri(:I,:D,:death)


epi_dict = Dict(:infection=>infection,
                :exposure=>exposure,
                :illness=>illness,
                :recovery=>recovery,
                :death=>death)

oapply_epi(ex, args...) = oapply(ex, epi_dict, args...)


Graph(infection)


Graph(recovery)


sir_wiring_diagram = @relation (s, i, r) begin
    infection(s, i)
    recovery(i, r)
end
display_uwd(sir_wiring_diagram)


sir_model = apex(oapply_epi(sir_wiring_diagram));
Graph(sir_model)


tmax = 40.0
tspan = (0.0,tmax);


u0 = LVector(S=990.0, I=10.0, R=0.0)


p = LVector(inf=0.05*10.0/sum(u0), rec=0.25); # β*c/N,γ


Random.seed!(1234);


prob_ode = ODEProblem(vectorfield(sir_model),u0,tspan,p)
sol_ode = solve(prob_ode, Tsit5());
plot(sol_ode)

