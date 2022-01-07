
using DifferentialEquations
using DelayDiffEq
using DiffEqCallbacks
using Tables
using DataFrames
using StatsPlots
using BenchmarkTools


function sir_dde!(du,u,h,p,t)
    (S,I,R) = u
    (β,c,τ) = p
    N = S+I+R
    infection = β*c*I/N*S
    (Sd,Id,Rd) = h(p, t-τ) # Time delayed variables
    Nd = Sd+Id+Rd
    recovery = β*c*Id/Nd*Sd
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
    end
    nothing
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]; # S,I.R


p = [0.05,10.0,4.0]; # β,c,τ


function sir_history(p, t)
    [1000.0, 0.0, 0.0]
end;


function affect_initial_recovery!(integrator)
    integrator.u[2] -= u0[2]
    integrator.u[3] += u0[2]

    reset_aggregated_jumps!(integrator)
end
cb_initial_recovery = DiscreteCallback((u,t,integrator) -> t == p[3], affect_initial_recovery!);


prob_dde = DDEProblem(DDEFunction(sir_dde!),
        u0,
        sir_history,
        tspan,
        p;
        constant_lags = [p[3]]);


alg = MethodOfSteps(Tsit5());


sol_dde = solve(prob_dde,alg, callback=cb_initial_recovery);


df_dde = DataFrame(Tables.table(sol_dde(t)'))
rename!(df_dde,["S","I","R"])
df_dde[!,:t] = t;


@df df_dde plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


function sir_ode_initial!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*c*I/N*S
        du[2] = β*c*I/N*S
        du[3] = 0
    end
    nothing
end;


prob_ode = ODEProblem(sir_ode_initial!,u0,(0,p[3]),p);
sol_ode = solve(prob_ode);
u1 = sol_ode[end]
u1[2] -= u0[2]
u1[3] += u0[2]


function ode_history(p, t, sol)
    sol(t)
end;
sir_history1(p,t)=ode_history(p,t,sol_ode)


prob_dde1 = DDEProblem(DDEFunction(sir_dde!),
        u1,
        sir_history1,
        (p[3],tmax),
        p;
        constant_lags = [p[3]]);
alg1 = MethodOfSteps(Tsit5());
sol_dde1 = solve(prob_dde1,alg1);


@benchmark solve(prob_dde, alg, callback=cb_initial_recovery)

