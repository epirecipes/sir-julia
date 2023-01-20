
using FractionalDiffEq
using Plots
using BenchmarkTools


function sir_ode!(du, u, p, t)
    (S, I, R) = u
    (β, γ, α) = p
    N = S+I+R
    du[1] = dS = -(β^α)*I/N*S
    du[2] = dI = (β^α)*I/N*S - (γ^α)*I
    du[3] = dR = (γ^α)*I
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);


u0 = [990.0, 10.0, 0.0];


α = 1.0
p = [0.5, 0.25, α];


prob_fode = FODESystem(sir_ode!, [α, α, α],  u0, tspan, p);


h = 0.1
sol_fode = solve(prob_fode, h, NonLinearAlg());


plot(sol_fode)


α₂ = 0.9
p₂ = [0.5, 0.25, α₂];
prob_fode₂ = FODESystem(sir_ode!, [α₂, α₂, α₂],  u0, tspan, p₂);
sol_fode₂ = solve(prob_fode₂, h, NonLinearAlg());
plot(sol_fode₂)


@benchmark solve(prob_fode, h, NonLinearAlg())

