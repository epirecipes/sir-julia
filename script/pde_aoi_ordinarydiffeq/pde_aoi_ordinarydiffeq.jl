
using OrdinaryDiffEq
using Plots;


function pde_mol!(du, u, p, t)

    # Extract discretization parameters
    Δa = p.Δa
    nI = p.nI

    # Extract states
    S = u[1]
    I = @view u[2:nI+1]
    R = u[end]
    N = S+sum(I)*Δa+R  # total population

    # Extract parameters
    βvec = p.βvec
    γvec = p.γvec

    # Compute force of infection
    λ = sum(βvec .* I * Δa)/N

    # Compute derivatives
    du[1] = dS = -λ * S                   # susceptibles

    # First age bin (j = 1)
    du[2] = -(I[1] - λ*S)/Δa - γvec[1]*I[1]

    # Remaining bins (j = 2...nI)
    @inbounds for j in 2:nI
        du[j+1] = -(I[j] - I[j-1])/Δa - γvec[j]*I[j]
    end

    # Removed compartment
    du[end] = dR = sum(γvec .* I * Δa) + I[nI]   # gain from recovery + ageing out
end;


β = 0.5
γ = 0.25                      
β_a(a) = β                    # infectiousness profile, β(a)
γ_a(a) = γ;                   # recovery rate γ(a)


tmax   = 40.0                  # Simulation time (days)
tspan  = (0.0, tmax)           # time span for the simulation
Δt     = 0.1                   # time step for the simulation output
amax   = 40.0                  # truncate infection age domain (days)
nI     = 400                  # number of age bins
Δa     = amax / nI            # bin width
ages   = range(Δa/2, stop=amax-Δa/2, length=nI)  # mid‑points
βvec   = β_a.(ages)           # vectorise kernels on grid
γvec   = γ_a.(ages);


p = (βvec=βvec, γvec=γvec, Δa=Δa, nI=nI);


S0            = 990.0          # susceptibles at t = 0
I0            = zeros(nI)      # infection‑age density
I0[1]         = 1000.0 - S0    # seed a small cohort in first bin
I0           /= Δa             # convert prevalence to (unnormalized) density
R0            = 0.0            # removed at t = 0, not reproductive number
u0 = vcat(S0, I0, R0);         # [S, I₁,...,Iₙ, R]


prob_pde_mol = ODEProblem(pde_mol!, u0, tspan, p)
sol_pde_mol = solve(prob_pde_mol, Tsit5(), saveat=Δt);


t_points = sol_pde_mol.t                   # time points
S_sol = sol_pde_mol[1, :]                  # susceptibles
I_sol = sol_pde_mol[2:end-1, :]            # infection-age density
Itotal_sol   = vec(sum(I_sol, dims=1)*Δa)  # total infected
R_sol = sol_pde_mol[end, :]                # removed
N_sol = S_sol .+ Itotal_sol .+ R_sol;      # total population


plot(t_points, S_sol, label="S", xlabel="Time", ylabel="Number")
plot!(t_points, Itotal_sol, label="ΣI")
plot!(t_points, R_sol, label="R")
plot!(t_points, N_sol, label="N")


function sir_ode!(du,u,p,t)
    (S,I,R) = u
    N = S+I+R
    (β,γ) = p
    @inbounds begin
        du[1] = -β*S*I/N
        du[2] = β*S*I/N - γ*I
        du[3] = γ*I
    end
    nothing
end
u0_ode = [S0, sum(I0)*Δa, R0]
p_ode = (β, γ)
prob_ode = ODEProblem(sir_ode!, u0_ode, tspan, p_ode)
sol_ode = solve(prob_ode, Tsit5(), saveat = Δt);


t_points_ode = sol_ode.t
S_sol_ode = sol_ode[1,:]
I_sol_ode = sol_ode[2,:]
R_sol_ode = sol_ode[3,:]
N_sol_ode = S_sol_ode + I_sol_ode + R_sol_ode;


l = @layout [a b;c d]
p1 = plot(t_points_ode, S_sol_ode, label="S ODE", xlabel="Time", ylabel="Number",lw=4,ls=:dot)
plot!(p1, t_points, S_sol, label="S PDE")
p2 = plot(t_points_ode, I_sol_ode, label="I ODE", xlabel="Time", ylabel="Number",lw=4,ls=:dot)
plot!(p2, t_points, Itotal_sol, label="I PDE")
p3 = plot(t_points_ode, R_sol_ode, label="R ODE", xlabel="Time", ylabel="Number",lw=4,ls=:dot)
plot!(p3, t_points, R_sol, label="R PDE")
p4 = plot(t_points_ode, N_sol_ode, label="N ODE", xlabel="Time", ylabel="Number",lw=4,ls=:dot,ylim=(999,1001))
plot!(p4, t_points, N_sol, label="N PDE")
plot(p1, p2, p3, p4, layout=l)


using DelayDiffEq
function sir_dde!(du,u,h,p,t)
    (S, I, R) = u
    N= S + I + R
    (β, γ, amax, S0, I0, R0) = p
    (Sd, Id, Rd) = h(p, t-amax)
    # β*Sd*Id/N = individuals infected at time t-amax
    # exp(-γ*amax) = probability of not recovering between t-amax and t
    outflow = exp(-γ*amax)*β*Sd*Id/N
    @inbounds begin
        du[1] = -β*S*I/N
        du[2] = β*S*I/N - γ*I - outflow
        du[3] = γ*I + outflow
    end
    nothing
end
# Assume that infectious individuals were introduced at time t=0 at the beginning of their infection
function sir_history(p, t)
    (β, γ, amax, S0, I0, R0) = p
     N = S0 + I0 + R0
    [N, 0.0, 0.0]
end
u0_dde = [S0, sum(I0)*Δa, R0] # Initial conditions the same as for the ODE model
p_dde = (β, γ, amax, S0, sum(I0)*Δa, R0) # History function needs the initial conditions
prob_dde = DDEProblem(DDEFunction(sir_dde!), u0_dde, sir_history, tspan, p_dde, constant_lags = [amax]);
sol_dde = solve(prob_dde, MethodOfSteps(Tsit5()), saveat = Δt);

