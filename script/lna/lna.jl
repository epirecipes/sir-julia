
using OrdinaryDiffEq
using ForwardDiff
using DataFrames
using LinearAlgebra
using StatsPlots


function sir_rates(u,p)
  (S,I,R) = u
  (β,c,γ) = p
  N = S+I+R
  infection = β*c*I/N*S
  recovery = γ*I
  [infection,recovery]
end;


sir_transitions = float.([[-1  1  0];
                          [ 0 -1  1]])


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]; # S,I.R
c0 = zeros(3,3) # covariance matrix
uc0 = vcat(diagm(0=>u0), c0)


p = [0.05,10.0,0.25]; # β,c,γ


rates = sir_rates
transitions = sir_transitions
nrates, nstates = size(transitions)


function ode(du, u, p, t)
    du[1:3] = transitions' * rates(u, p)
end


prob_ode = ODEProblem(ode, u0 ,tspan, p)


sol_ode = solve(prob_ode, Tsit5(), saveat=t)


df_ode = DataFrame(sol_ode(t)')
df_ode[!,:t] = t;


@df df_ode plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number",
    c=[:blue :red :green])


function lna(du, u, p, t)
        mean_vec = diag(u)
        covar_mx = u[nstates+1:nstates*2,:]
        reaction_rates = rates(mean_vec, p)
        reaction_rates_jac = ForwardDiff.jacobian(
            y -> rates(y, p),
            mean_vec)
        A = transitions' * reaction_rates_jac
        du[1:nstates, :] .= diagm(0 => transitions'*reaction_rates)
        du[nstates + 1:end, :] .= A*covar_mx + covar_mx*A' + transitions' * diagm(0 => reaction_rates) * transitions
end


prob_lna = ODEProblem(lna, uc0, tspan, p)


sol_lna = solve(prob_lna, Tsit5(),saveat=t)


mean_traj = Array{Float64,2}(undef, nstates, length(t))
covar_traj = Array{Array{Float64,2},1}(undef, length(t))
for j in 1:length(t)
    mean_traj[:,j] = diag(sol_lna[j][1:nstates,1:nstates])
    covar_traj[j] = sol_lna[j][nstates+1:end, 1:nstates]
end


var_traj = zeros(nstates, length(t))
for (idx, elt) in enumerate(covar_traj)
    var_traj[:, idx] = diag(elt)
end
sd_traj = 1.96 .* sqrt.(var_traj)


plot(t, mean_traj',
     ribbon=sd_traj',
     label = ["S" "I" "R"],
     xlabel = "Time",
     ylabel = "Number")


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

