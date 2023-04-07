
using OrdinaryDiffEq
using DiffEqCallbacks
using DiffEqSensitivity
using Random
using Distributions
using DataInterpolations
using DynamicHMC
using Turing
using Optim
using LinearAlgebra
using DataFrames
using StatsBase
using StatsPlots


function sir_ode!(du,u,p,t)
    (S, I, C) = u
    (β, γ, N) = p
    infection = β*S*I/N
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = infection
    end
    nothing
end;


lockdown_times = [10.0, 20.0]
condition(u,t,integrator) = t ∈ lockdown_times
function affect!(integrator)
    if integrator.t < lockdown_times[2]
        integrator.p[1] = 0.1
    else
        integrator.p[1] = 0.5
    end
end
cb = PresetTimeCallback(lockdown_times, affect!);


tmax = 40.0
tspan = (0.0, tmax)
obstimes = 1.0:1.0:tmax
u0 = [990.0, 10.0, 0.0] # S,I,C
N = 1000.0 # Population size
p = [0.5, 0.25, N]; # β, γ, N


prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode = solve(prob_ode,
            Tsit5(),
            callback = cb,
            saveat = 1.0);


plot(sol_ode,
    xlabel="Time",
    ylabel="Number",
    labels=["S" "I" "C"])


C = [0; Array(sol_ode(obstimes))[3,:]] # Cumulative cases
X = C[2:end] - C[1:(end-1)];


Random.seed!(1234)
Y = rand.(Poisson.(X));


bar(obstimes, Y, legend=false)
plot!(obstimes, X, legend=false)


true_beta = [0.5, 0.1, 0.5, 0.5, 0.5]
knots = collect(0.0:10.0:tmax)
K = length(knots)
function betat(p_, t)
    beta = ConstantInterpolation(p_, knots)
    return beta(t)
end;


function sir_tvp_ode!(du, u, p_, t)
    (S, I, C) = u
    (_, γ, N) = p
    βt = betat(p_, t)
    infection = βt*S*I/N
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = infection
    end
    nothing
end;


prob_tvp = ODEProblem(sir_tvp_ode!,
          u0,
          tspan,
          true_beta);


@model bayes_sir_tvp(y, K) = begin
  # Set prior for initial infected
  i₀  ~ Uniform(0.0, 0.1)
  I = i₀*N
  u0 = [N-I, I, 0.0]
  # Set priors for betas
  ## Note how we clone the endpoint of βt
  βt = Vector{Float64}(undef, K)
  for i in 1:K-1
    βt[i] ~ Uniform(0.0, 1.0)
  end
  βt[K] = βt[K-1]
  # Run model
  ## Remake with new initial conditions and parameter values
  prob = remake(prob_tvp,
          u0=u0,
          p=βt)
  ## Solve
  sol = solve(prob,
              Tsit5(),
              saveat = 1.0)
  ## Calculate cases per day, X
  sol_C = [0; Array(sol(obstimes))[3,:]]
  sol_X = abs.(sol_C[2:end] - sol_C[1:(end-1)])
  # Assume Poisson distributed counts
  ## Calculate number of timepoints
  l = length(y)
  for i in 1:l
    y[i] ~ Poisson(sol_X[i])
  end
end;


advi = ADVI(10, 1000) # 10 samples, 1000 gradient iterations
@time ode_advi = vi(bayes_sir_tvp(Y, K), advi);


ode_advi_postsamples = rand(ode_advi, 1000);


beta_idx = [collect(2:K);K]
betas = [mean(ode_advi_postsamples[i,:]) for i in beta_idx]
betas_lci = [quantile(ode_advi_postsamples[i,:], 0.025) for i in beta_idx]
betas_uci = [quantile(ode_advi_postsamples[i,:], 0.975) for i in beta_idx];


plot(obstimes,
     betat(betas, obstimes),
     xlabel = "Time",
     ylabel = "β",
     label="Estimated β",
     title="ADVI estimates",
     color=:blue)
plot!(obstimes,
     betat(betas_lci, obstimes),
     alpha = 0.3,
     fillrange = betat(betas_uci, obstimes),
     fillalpha = 0.3,
     color=:blue,
     label="95% credible intervals")
plot!(obstimes,
     betat(true_beta, obstimes),
     color=:red,
     label="True β")


histogram(ode_advi_postsamples[1,:],
    xlabel="Fraction of initial infected",
    normed=true,
    alpha=0.2,
    color=:blue,
    label="",
    title="HMC estimate")
density!(ode_advi_postsamples[1,:], color=:blue, label="")
vline!([0.01], color=:red, label="True value")


burnin = 1000
nchains = 4
samples = 50000;


@time ode_hmc = sample(bayes_sir_tvp(Y, K),
                  HMC(0.025, 10),
                  MCMCThreads(),
                  burnin+samples,
                  nchains);


ode_hmc_description = describe(ode_hmc[(burnin+1):end,:,:])
ode_hmc_description[1]


ode_hmc_description[2]


plot(ode_hmc[(burnin+1):end,:,:])


betas_hmc = ode_hmc_description[1][:,2][beta_idx]
betas_hmc_lci = ode_hmc_description[2][:,2][beta_idx]
betas_hmc_uci = ode_hmc_description[2][:,6][beta_idx];


plot(obstimes,
     betat(betas_hmc, obstimes),
     xlabel = "Time",
     ylabel = "β",
     label="Estimated β",
     title="HMC estimates",
     color=:blue)
plot!(obstimes,
     betat(betas_hmc_lci, obstimes),
     alpha = 0.3,
     fillrange = betat(betas_hmc_uci, obstimes),
     fillalpha = 0.3,
     color=:blue,
     label="95% credible intervals")
plot!(obstimes,
     betat(true_beta, obstimes),
     color=:red,
     label="True β")


histogram(ode_hmc[(burnin+1):end,1,1],
    xlabel="Fraction of initial infected",
    normed=true,
    alpha=0.2,
    color=:blue,
    label="",
    title="HMC estimate")
density!(ode_hmc[(burnin+1):end,1,1], color=:blue, label="")
vline!([0.01], color=:red, label="True value")

