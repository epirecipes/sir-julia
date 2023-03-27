
using OrdinaryDiffEq
using Random
using Distributions
using StatsBase
using ThreadsX
using Plots
using Loess


function sir_markov(u,p,t)
    (S, I, _) = u
    C = 0
    (β, γ, N) = p
    δt = 0.1
    nsteps=10
    for i in 1:nsteps
        ifrac = 1-exp(-β*I/N*δt)
        rfrac = 1-exp(-γ*δt)
        infection = rand(Binomial(S,ifrac))
        recovery = rand(Binomial(I,rfrac))
        S = S-infection
        I = I+infection-recovery
        C = C+infection
    end
   [S, I, C]
end;


tspan = (0,40)
u0 = [990, 10, 0] # S, I, C
β = 0.5
γ = 0.25
N = 1000
p = [β, γ, N] # β, γ, N
seed = 1234
nparticles = 100000;


Random.seed!(seed);


prob = DiscreteProblem(sir_markov, u0, tspan, p, dt=1)
sol = solve(prob, FunctionMap())
C = hcat(sol.u...)[3,2:end];


function pfilter(prob, p, u0, C, nparticles=nparticles, seed=seed)
    # Remake with parameters and initial conditions
    prob = remake(prob, p=p, u0=u0)
    # Generate a vector of integrators
    integrators = [init(prob, FunctionMap(),save_everystep=true) for i in 1:nparticles]
    # Initialize
    Random.seed!(seed)
    liks = zeros(Float64,length(C))
    weights = Weights(zeros(Float64,nparticles))
    us = [copy(u0) for i in 1:nparticles]
    idx = collect(1:nparticles)
    # Filter each timepoint
    @inbounds for t in 1:length(C)
        step!.(integrators) # Take a step
        c = C[t] # The data at t
        [us[i] = integrators[i].u for i in 1:nparticles]
        [weights[i]=Float64(us[i][3]==c) for i in 1:nparticles] # 1.0 if state==c, 0.0 otherwise
        liks[t] = mean(weights)
        # Some naive failure handling
        if mean(weights)==0.0
            return -Inf
            break
        end
        # Resample indices according to weights
        sample!(1:nparticles, weights, idx)
        # Reinitialize integrators with resampled states
        [reinit!(integrators[i],us[idx[i]]) for i in 1:nparticles]
    end
    sum(log.(liks))
end;


pfilter(prob, p, u0, C, nparticles, seed)


## Array of β values
betas = collect(0.35:0.005:0.7)
# Use ThreadsX to parallelise across parameter runs
@time beta_liks = ThreadsX.collect(pfilter(prob, [beta, γ, N], u0, C, nparticles, seed) for beta in betas);


betas_failed = beta_liks.==-Inf
betas_success = betas[.!betas_failed]
beta_liks_success = beta_liks[.!betas_failed]
betas_model = loess(betas_success, beta_liks_success)
beta_liks_smooth = Loess.predict(betas_model, betas_success)
β̂=betas_success[argmax(beta_liks_smooth)]


plot(betas_success,
    beta_liks_smooth,
    xlabel="β",
    ylabel="Log likelihood",
    label="",
    legend=true,
    marker=false)
scatter!(betas, beta_liks, label="")
vline!([p[1]],label="True β")
vline!([β̂],label="Estimated β")


I0s = collect(1:20)
@time I0s_liks = ThreadsX.collect(pfilter(prob, [β, γ, N], [N-I0, I0, 0], C, 2*nparticles, seed) for I0 in I0s);


I0s_failed = I0s_liks.==-Inf
I0s_success = I0s[.!I0s_failed]
I0s_liks_success = I0s_liks[.!I0s_failed]
Î₀ = I0s_success[argmax(I0s_liks_success)]


plot(I0s,
    I0s_liks,
    xlabel="I₀",
    ylabel="Log likelihood",
    label="",
    legend=true,
    marker=true,
    xtick=I0s)
vline!([u0[2]],label="True I₀")
vline!([Î₀],label="Estimated I₀")

