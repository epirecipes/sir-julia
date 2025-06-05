
using OrdinaryDiffEq
using DiffEqCallbacks
using Interpolations
using Distributions
using Random
using DataFrames
using Plots


function sir_ode!(du,u,p,t)
    (S,I,C) = u
    (β,γ) = p
    @inbounds begin
        du[1] = -β*S*I
        du[2] = β*S*I - γ*I
        du[3] = β*S*I
    end
    nothing
end;


tspan = (0.0, 10000.0)
dt = 0.1
u0 = [0.99, 0.01, 0.0]
p = [0.5, 0.25];


prob = ODEProblem(sir_ode!, u0, tspan, p)
cb = TerminateSteadyState(1e-8)
sol = solve(prob, Tsit5(); dt=dt, dtmax=dt, callback=cb);


plot(sol,
     title="SIR Model",
     xlabel="Time",
     ylabel="Population Fraction",
     label=["Susceptible" "Infected" "Cumulative Infected"])


τ = sol[end][3] # Final size
times = sol.t # Simulation times
cdfτ = [sol[i][3]/τ for i in 1:length(sol.t)] # CDF evaluated at `times`, obtained by C/τ
invcdfτ = LinearInterpolation(cdfτ, times, extrapolation_bc=Line());


plot(times, cdfτ,
     title="Cumulative Density Function",
     xlabel="Time",
     ylabel="Density of infection times",
     label=false)


N = 99 # Sample N initially susceptible
M = 1 # Sample M initially infected
K = rand(Binomial(N, τ)) # Number of infections during the epidemic
Tᵢ = zeros(M) # Assume times of infection for the M initially infected individuals are zero
Tᵢ = [Tᵢ; invcdfτ.(rand(K))] # Append infection times for K individuals
Tᵣ = Tᵢ .+ rand(Exponential(1/p[2]), M+K); # Recovery times for M+K infected individuals


df = DataFrame(infection_time=Tᵢ, recovery_time=Tᵣ)

