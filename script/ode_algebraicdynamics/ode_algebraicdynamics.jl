
using AlgebraicDynamics
using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.CPortGraphDynam
using AlgebraicDynamics.CPortGraphDynam: barbell
using Catlab.WiringDiagrams
using Catlab.Programs # for @relation macro
using Catlab.Graphics # for to_graphviz
using OrdinaryDiffEq
using DataFrames
using Plots
using StatsPlots


δt = 0.1
tmax = 40
tspan = (0.0,tmax)
t = 0:δt:tmax;


u0 = [990.0,10.0];


β, γ = 0.05*10/1000, 0.25; # in other tutorials, βc/N and γ


dots(u, x, p, t) = [-β*u[1]*x[1]]


doti(u, x, p, t) = [β*x[1]*u[1] - γ*u[1]];


susceptible_cm = ContinuousMachine{Float64}(1,1,1, dots, u -> u)
infected_cm    = ContinuousMachine{Float64}(1,1,1, doti, u -> u);


directed_pattern = WiringDiagram([], [])
susceptible_box = add_box!(directed_pattern, Box(:susceptible_cm, [:pop], [:pop]))
infected_box = add_box!(directed_pattern, Box(:infected_cm, [:pop], [:pop]));


add_wires!(directed_pattern, Pair[
    (susceptible_box, 1) => (infected_box, 1),
    (infected_box, 1) => (susceptible_box, 1)
])


to_graphviz(directed_pattern)


directed_system = oapply(directed_pattern, [susceptible_cm, infected_cm]);


directed_prob = ODEProblem(directed_system, u0, tspan)
directed_sol = solve(directed_prob, FRK65(0));


plot(directed_sol)


cpg_pattern = barbell(1)


cpg_system = oapply(cpg_pattern, [susceptible_cm, infected_cm]);


cpg_prob = ODEProblem(cpg_system, u0, tspan)
cpg_sol = solve(cpg_prob, FRK65(0));


plot(cpg_sol)


dotsi(u,p,t) = [-β*u[1]*u[2],β*u[1]*u[2]]


doti(u,p,t) = -γ*u;


si_infection = ContinuousResourceSharer{Float64}(2, dotsi)
i_recovery = ContinuousResourceSharer{Float64}(1, doti);


undirected_pattern = @relation (S, I) begin
    si_infection(S,I)
    i_recovery(I)
end


to_graphviz(undirected_pattern, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))


undirected_system = oapply(undirected_pattern, [si_infection, i_recovery]);


undirected_prob = ODEProblem(undirected_system, u0, tspan)
undirected_sol = solve(undirected_prob,FRK65(0));


plot(undirected_sol)


μ = 1.0/10
dotis(u,p,t) = [-μ*u[1],μ*u[1]];


is_birthdeath = ContinuousResourceSharer{Float64}(2, dotis);


undirected_open_pattern = @relation (S, I) begin
    si_infection(S,I)
    i_recovery(I)
    is_birthdeath(I,S)
end


to_graphviz(undirected_open_pattern, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))


undirected_open_system = oapply(undirected_open_pattern, [si_infection, i_recovery, is_birthdeath]);


undirected_open_prob = ODEProblem(undirected_open_system, u0, tspan)
undirected_open_sol = solve(undirected_open_prob,FRK65(0));


plot(undirected_open_sol)


nstages = 4
δ = nstages*γ;


sub(i::Int) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))
sub(x::String,i::Int) = x*sub(i)
istages = [sub("I",i) for i=1:nstages]


dotsii(u,p,t) = [-β*u[1]*u[3],β*u[1]*u[3],0.0]
dotii(u,p,t) = [-δ*u[1],δ*u[1]]
dotilast(u,p,t) = [-δ*u[1]];


sii_infection = ContinuousResourceSharer{Float64}(3, dotsii)
i_transition = ContinuousResourceSharer{Float64}(2, dotii)
ilast_recovery = ContinuousResourceSharer{Float64}(1, dotilast);


undirected_pattern_stages = @relation (S, I₁, I₂, I₃, I₄) begin
    si_infection(S,I₁)
    sii_infection(S,I₁,I₂)
    sii_infection(S,I₁,I₃)
    sii_infection(S,I₁,I₄)
    i_transition(I₁,I₂)
    i_transition(I₂,I₃)
    i_transition(I₃,I₄)
    ilast_recovery(I₄)
end


to_graphviz(undirected_pattern_stages, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))


undirected_system_stages = oapply(undirected_pattern_stages, [
    si_infection
    sii_infection
    sii_infection
    sii_infection
    i_transition
    i_transition
    i_transition
    ilast_recovery])


undirected_system_stages = oapply(undirected_pattern_stages, Dict(
    :si_infection  => si_infection,
    :sii_infection => sii_infection,
    :i_transition  => i_transition,
    :ilast_recovery => ilast_recovery
))


u0stages = [990.0,10.0,0.0,0.0,0.0];


undirected_stages_prob = ODEProblem(undirected_system_stages, u0stages, tspan)
undirected_stages_sol = solve(undirected_stages_prob,FRK65(0));


undirected_stages_df = DataFrame(undirected_stages_sol(t)')
rename!(undirected_stages_df,["S";istages])
undirected_stages_df[!,:I] = undirected_stages_df[!,:I₁] +
                              undirected_stages_df[!,:I₂] +
                              undirected_stages_df[!,:I₃] +
                              undirected_stages_df[!,:I₄]
undirected_stages_df[:t] = t;


plot(undirected_stages_df[!,:t],
     [undirected_stages_df[!,:S],undirected_stages_df[!,:I]])
plot!(undirected_sol)


undirected_si_pattern = @relation (S, I₁, I₂, I₃, I₄) begin
    si_box(S, I₁, I₂, I₃, I₄)
    i_box(I₁, I₂, I₃, I₄)
end;


to_graphviz(undirected_si_pattern, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))


si_pattern = @relation (S, I₁, I₂, I₃, I₄) begin
    si_infection(S,I₁)
    sii_infection(S,I₁,I₂)
    sii_infection(S,I₁,I₃)
    sii_infection(S,I₁,I₄)
end;


to_graphviz(si_pattern, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))


i_pattern = @relation (I₁, I₂, I₃, I₄) begin
    i_transition(I₁,I₂)
    i_transition(I₂,I₃)
    i_transition(I₃,I₄)
    ilast_recovery(I₄)
end;


to_graphviz(i_pattern, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))


undirected_pattern_stages = ocompose(undirected_si_pattern, [si_pattern, i_pattern]);


to_graphviz(undirected_pattern_stages, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))

