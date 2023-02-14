
using FdeSolver
using Plots
using BenchmarkTools


function sir_ode(t, u, p)
    (S, I, R) = u
    (β, γ, α) = p
    N = S+I+R
    dS = -(β^α)*I/N*S
    dI = (β^α)*I/N*S - (γ^α)*I
    dR = (γ^α)*I
    [dS, dI, dR]
end;


tspan = [0.0, 40.0];
u0 = [990.0, 10.0, 0.0];
α = 0.9
p = [0.5, 0.25, α];


t, sol_fode = FDEsolver(sir_ode, tspan, u0, [α, α, α], p, h = 0.1);


plot(t, sol_fode)


α₂ = 0.9
p₂ = [0.5, 0.25, α₂];
t₂, sol_fode₂ = FDEsolver(sir_ode, tspan, u0, [α₂, α₂, α₂], p₂, h = 0.1);
plot(sol_fode₂)


@benchmark FDEsolver(sir_ode, tspan, u0, [α, α, α], p, h = 0.1)

