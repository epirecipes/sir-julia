
using DifferentialEquations
using SimpleDiffEq
using Plots
using BenchmarkTools


function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*S*I/N
        du[2] = β*S*I/N - γ*I
        du[3] = γ*I
    end
    nothing
end;


tspan = (0.0,40.0)
u0 = [990.0,10.0,0.0]
p = [0.5,0.25];


prob_sir_ode = ODEProblem(sir_ode!,u0,tspan,p)


sol_sir_ode = solve(prob_sir_ode);


plot(sol_sir_ode,vars=[(0,1),(0,2),(0,3)])


@benchmark solve(prob_sir_ode)

