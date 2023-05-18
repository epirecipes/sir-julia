
using StockFlow
using LabelledArrays
using OrdinaryDiffEq
using GraphViz
using Plots;


fInfection(u, p, t) = p.β * p.c * u.S * u.I/(u.S + u.I + u.R)
fRecovery(u, p, t) = p.γ * u.I;


sirp = StockAndFlowp((:S, :I, :R), 
   ((:Infection =>  fInfection, :S=>:I) =>  (:S,:I),
    (:Recovery  =>  fRecovery, :I=>:R)  =>  :I)
)


Graph(sirp)


vfp = vectorfield(sirp);


tmax = 40.0
tspan = (0, tmax)
δt = 0.1;


u0 = LVector(S=990, I=10, R=0);


p = LVector(β=0.05, c=10, γ=0.25);


prob_p = ODEProblem(vfp, u0, tspan, p)
sol_p = solve(prob_p, Tsit5(), saveat=δt);


plot(sol_p, xlabel="Time", ylabel="Number")


fInfection(u, uN, p, t) = p.β * p.c * u.S * u.I/uN.N(u,t)
fRecovery(u, uN, p, t) = p.γ * u.I;


sir=StockAndFlow(
    (:S=>(:F_NONE, :infection , :v_infection, :N),
     :I=>(:infection, :recovery, (:v_infection, :v_recovery), :N),
     :R=>(:recovery, :F_NONE, :V_NONE, :N)),
    (:infection => :v_infection, :recovery => :v_recovery),
    (:v_infection => fInfection, :v_recovery => fRecovery),
    (:N => (:v_infection))
)


vf = vectorfield(sir)
prob = ODEProblem(vf, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=δt);


plot(sol, xlabel="Time", ylabel="Number")


Graph(sir)


sir_causalloop = convertToCausalLoop(sir)


Graph(sir_causalloop)


sir_structure = convertStockFlowToSystemStructure(sir)


Graph(sir_structure)

