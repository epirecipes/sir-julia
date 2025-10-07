
using Turing
using MCMCChains
using Distributions
using Random
using Plots
using StatsPlots
using Base.Threads;


include("generalized_binomial.jl")
import .GeneralizedBinomialExt: GeneralizedBinomial;


@inline function rate_to_proportion(r,t)
    1-exp(-r*t)
end;


function sir_map!(du,u,p,t)
    (S, I, R, C) = u
    (β, γ, q, N, δt) = p
    infection = rate_to_proportion(β*I/N, δt)*S
    recovery = rate_to_proportion(γ, δt)*I
    @inbounds begin
        du[1] = S - infection
        du[2] = I + infection - recovery
        du[3] = R + recovery
        du[4] = C + infection
    end
    nothing
end;


function solve_map(f, u0, nsteps, p)
    # Pre-allocate array with correct type
    sol = similar(u0, length(u0), nsteps + 1)
    # Initialize the first column with the initial state
    sol[:, 1] = u0
    # Iterate over the time steps
    @inbounds for t in 2:nsteps+1
        u = @view sol[:, t-1] # Get the current state
        du = @view sol[:, t]  # Prepare the next state
        f(du, u, p, t)        # Call the function to update du
    end
    return sol
end;


δt = 1.0 # Time step
nsteps = 40
tmax = nsteps*δt
t = 0.0:δt:tmax;


u0 = [990.0, 10.0, 0.0, 0.0];


p = (β=0.5, γ=0.25, q=0.75, N=1000.0, δt=δt);


sol_map = solve_map(sir_map!, u0, nsteps, p);


S, I, R, C = eachrow(sol_map);


plot(t,
     [S I R C],
     label=["S" "I" "R" "C"],
     xlabel="Time",
     ylabel="Number")


Y = rand.(GeneralizedBinomial.(C[2:end]-C[1:end-1], p.q));


ZN = 100
Z = rand(GeneralizedBinomial(ZN, R[end]/p.N));


function logpdf(Y, sol, q)
    C = sol[4,:]
    ll = 0.0
    X = (C[2:end] .- C[1:end-1])
    for i in 1:length(Y)
        ll += Distributions.logpdf(GeneralizedBinomial(X[i], q), Y[i])
    end
    return ll
end;


@model function sir_map_estimate_q(Y, u0, nsteps, p)
    # Priors for the parameters we want to estimate
    β ~ Uniform(0.25, 0.75)
    I₀ ~ Uniform(5.0, 50.0)
    q ~ Uniform(0.1, 0.9)

    # Create parameter tuple with current MCMC values
    p_new = merge(p, (β = β, q = q))
    u0_new = [p.N - I₀, I₀, 0.0, 0.0]

    # Solve the model with the current parameters
    sol = solve_map(sir_map!, u0_new, nsteps, p_new)

    # Add the log-likelihood of the cases to the model
    Turing.@addlogprob! logpdf(Y, sol, q)

    return nothing
end;


sir_model_estimate_q = sir_map_estimate_q(Y, u0, nsteps, p)
chain_estimate_q = sample(sir_model_estimate_q, NUTS(0.65), 10000; progress=false);


describe(chain_estimate_q)


plot(chain_estimate_q)


nsims = 1000
I₀_means = Array{Float64}(undef, nsims)
β_means = Array{Float64}(undef, nsims)
q_means = Array{Float64}(undef, nsims)
I₀_coverage = Array{Float64}(undef, nsims)
β_coverage = Array{Float64}(undef, nsims)
q_coverage = Array{Float64}(undef, nsims)
Threads.@threads for i in 1:nsims
    Y_sim = rand.(GeneralizedBinomial.(C[2:end]-C[1:end-1], p.q))
    r = sample(sir_map_estimate_q(Y_sim, u0, nsteps, p),
               NUTS(1000,0.65),
               10000;
               verbose=false,
               progress=false,
               initial_params=(β=0.5, I₀=10.0, q=0.75))
    I₀_means[i] = mean(r[:I₀])
    I₀_cov = sum(r[:I₀] .<= u0[2]) / length(r[:I₀])
    β_means[i] = mean(r[:β])
    β_cov = sum(r[:β] .<= p.β) / length(r[:β])
    q_means[i] = mean(r[:q])
    q_cov = sum(r[:q] .<= p.q) / length(r[:β])
    I₀_coverage[i] = I₀_cov
    β_coverage[i] = β_cov
    q_coverage[i] = q_cov
end;


# Convenience function to check if the true value is within the credible interval
function in_credible_interval(x, lwr=0.025, upr=0.975)
    return x >= lwr && x <= upr
end;


pl_β_coverage = histogram(β_coverage, bins=0:0.1:1.0, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
pl_I₀_coverage = histogram(I₀_coverage, bins=0:0.1:1.0, label=false, title="i₀", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
pl_q_coverage = histogram(q_coverage, bins=0:0.1:1.0, label=false, title="q", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
plot(pl_β_coverage, pl_I₀_coverage, pl_q_coverage, layout=(1,3), plot_title="Distribution of CDF of true value")


sum(in_credible_interval.(β_coverage)) / nsims


sum(in_credible_interval.(I₀_coverage)) / nsims


sum(in_credible_interval.(q_coverage)) / nsims


pl_β_means = histogram(β_means, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.48, 0.52))
vline!([p.β], label="True value")
pl_I₀_means = histogram(I₀_means, label=false, title="I₀", ylabel="Density", density=true, xrotation=45, xlim=(5.0,15.0))
vline!([u0[2]], label="True value")
pl_q_means = histogram(q_means, label=false, title="q", ylabel="Density", density=true, xrotation=45, xlim=(0.65,0.85))
vline!([p.q], label="True value")
plot(pl_β_means, pl_I₀_means, pl_q_means, layout=(1,3), plot_title="Distribution of posterior means")


@model function sir_map_estimate_q_prevalence(Y, Z, ZN, u0, nsteps, p)
    # Priors for the parameters we want to estimate
    β ~ Uniform(0.25, 0.75)
    I₀ ~ Uniform(5.0, 50.0)
    q ~ Uniform(0.1, 0.9)

    # Create parameter tuple with current MCMC values
    p_new = merge(p, (β = β, q = q))
    u0_new = [p.N - I₀, I₀, 0.0, 0.0]

    # Solve the model with the current parameters
    sol = solve_map(sir_map!, u0_new, nsteps, p_new)

    # Add the log-likelihood of the cases to the model
    Turing.@addlogprob! logpdf(Y, sol, q)
    
    # Calculate contribution from end prevalence study
    zp = sol[3,end]/p.N
    zp = max(min(zp,1.0),0.0) # To ensure boundedness
    Z ~ GeneralizedBinomial(ZN, zp)

    return nothing
end;


sir_model_estimate_q_prevalence = sir_map_estimate_q_prevalence(Y, Z, ZN, u0, nsteps, p)
chain_estimate_q_prevalence = sample(sir_model_estimate_q_prevalence, NUTS(0.65), 10000; progress=false);


describe(chain_estimate_q_prevalence)


plot(chain_estimate_q_prevalence)


I₀_prev_means = Array{Float64}(undef, nsims)
β_prev_means = Array{Float64}(undef, nsims)
q_prev_means = Array{Float64}(undef, nsims)
I₀_prev_coverage = Array{Float64}(undef, nsims)
β_prev_coverage = Array{Float64}(undef, nsims)
q_prev_coverage = Array{Float64}(undef, nsims)
Threads.@threads for i in 1:nsims
    Y_sim = rand.(GeneralizedBinomial.(C[2:end]-C[1:end-1], p.q))
    Z_sim = rand(GeneralizedBinomial(ZN, R[end]/p.N))
    r = sample(sir_map_estimate_q_prevalence(Y_sim, Z_sim, ZN, u0, nsteps, p),
               NUTS(1000,0.65),
               10000;
               verbose=false,
               progress=false,
               initial_params=(β=0.5, I₀=10.0, q=0.75))
    I₀_prev_means[i] = mean(r[:I₀])
    I₀_cov = sum(r[:I₀] .<= u0[2]) / length(r[:I₀])
    β_prev_means[i] = mean(r[:β])
    β_cov = sum(r[:β] .<= p.β) / length(r[:β])
    q_prev_means[i] = mean(r[:q])
    q_cov = sum(r[:q] .<= p.q) / length(r[:q])
    I₀_prev_coverage[i] = I₀_cov
    β_prev_coverage[i] = β_cov
    q_prev_coverage[i] = q_cov
end;


pl_β_prev_coverage = histogram(β_prev_coverage, bins=0:0.1:1.0, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
pl_I₀_prev_coverage = histogram(I₀_prev_coverage, bins=0:0.1:1.0, label=false, title="i₀", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
pl_q_prev_coverage = histogram(q_prev_coverage, bins=0:0.1:1.0, label=false, title="q", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
plot(pl_β_prev_coverage, pl_I₀_prev_coverage, pl_q_prev_coverage, layout=(1,3), plot_title="Distribution of CDF of true value")


sum(in_credible_interval.(β_prev_coverage)) / nsims


sum(in_credible_interval.(I₀_prev_coverage)) / nsims


sum(in_credible_interval.(q_prev_coverage)) / nsims


pl_β_prev_means = histogram(β_prev_means, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.48, 0.52))
vline!([p.β], label="True value")
pl_I₀_prev_means = histogram(I₀_prev_means, label=false, title="I₀", ylabel="Density", density=true, xrotation=45, xlim=(5.0,15.0))
vline!([u0[2]], label="True value")
pl_q_prev_means = histogram(q_prev_means, label=false, title="q", ylabel="Density", density=true, xrotation=45, xlim=(0.65,0.85))
vline!([p.q], label="True value")
plot(pl_β_prev_means, pl_I₀_prev_means, pl_q_prev_means, layout=(1,3), plot_title="Distribution of posterior means")

