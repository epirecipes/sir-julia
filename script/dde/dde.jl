
using DifferentialEquations
using DelayDiffEq
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


function sir_history(p, t)
    [1000.0, 0.0, 0.0]
end;


p = [0.05,10.0,4.0]; # β,c,τ


prob_dde = DDEProblem(DDEFunction(sir_dde!),
        u0,
        sir_history,
        tspan,
        p;
        constant_lags = [p[3]]);


alg = MethodOfSteps(Tsit5());


sol_dde = solve(prob_dde,alg);


df_dde = DataFrame(sol_dde(t)')
df_dde[!,:t] = t;


@df df_dde plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_dde, alg)

