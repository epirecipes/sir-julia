
using OrdinaryDiffEq
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
using Random
using LinearAlgebra # Provides identity matrix `I`
using Distributions
using ThreadsX
using Plots
using StatsPlots;


function sir_markov(u,p,t)
    (S, I, C) = u
    C = 0
    (β, γ, N) = p
    δt = 0.1
    nsteps = 10
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
i₀ = 0.01
p = [β, γ, N]
seed = 1234;


Random.seed!(seed)
prob = DiscreteProblem(sir_markov, u0, tspan, p)
sol = solve(prob, FunctionMap())
plot(sol, labels=["S" "I" "C"], xlabel="Time", ylabel="Number")


C = Float64.(hcat(sol.u...)[3,2:end])
summary_stats = [maximum(C), Float64(argmax(C)),  sum(C)]
cases = log.(C .+ 1);


function get_summary_stats(q)
    i0 = Float64(round(N*q[2]))
    problem = remake(prob, p=[q[1], γ, N],u0=[N-i0,i0,0.0])
    sol = solve(problem, FunctionMap())
    C = Float64.(hcat(sol.u...)[3,2:end])
    return [maximum(C), Float64(argmax(C)),  sum(C)]
end;


function get_cases(q)
    i0 = Float64(round(N*q[2]))
    problem = remake(prob, p=[q[1], γ, N],u0=[N-i0,i0,0.0])
    sol = solve(problem, FunctionMap())
    C = Float64.(hcat(sol.u...)[3,2:end])
    return log.(C .+ 1)
end;


prior_u1 = constrained_gaussian("β", 0.5, 0.3, 0.0, 1.0)
prior_u2 = constrained_gaussian("i₀", 0.05, 0.03, 0.0, 0.1)
prior = combine_distributions([prior_u1, prior_u2]);


Γ = 1e-4 * LinearAlgebra.I
prior_mean = mean(prior)
prior_cov = cov(prior)
eks_process = Sampler(prior_mean, prior_cov);


N_iterations_ss = 50
N_ensemble_ss = 1000
rng_ss = Random.Xoshiro(seed)
initial_ensemble_ss = EKP.construct_initial_ensemble(rng_ss, prior, N_ensemble_ss);


eks_obj_ss = EKP.EnsembleKalmanProcess(initial_ensemble_ss, summary_stats, Γ, eks_process; rng=rng_ss)
for i in 1:N_iterations_ss
    params_i = get_ϕ_final(prior, eks_obj_ss)
    # Without threads would be as follows
    # ss = hcat([get_summary_stats(params_i[:, i]) for i in 1:N_ensemble_ss]...)
    ss = hcat(ThreadsX.collect(get_summary_stats(params_i[:, i]) for i in 1:N_ensemble_ss)...)
    EKP.update_ensemble!(eks_obj_ss, ss)
end;


θ_post_ss = get_u_mean_final(eks_obj_ss)
Γ_post_ss = get_u_cov_final(eks_obj_ss);


post_samples_ss_unconstrained = rand(MvNormal(θ_post_ss, Γ_post_ss), 10000)
post_samples_ss = transform_unconstrained_to_constrained(prior, post_samples_ss_unconstrained);


quantile(post_samples_ss[1,:], [0.025, 0.5, 0.975])


quantile(post_samples_ss[2,:], [0.025, 0.5, 0.975])


corrplot(post_samples_ss',labels=["β" "i₀"])


prior_ensemble_ss = get_ϕ(prior, eks_obj_ss, 1);


l = @layout [a b; c d]
p1 = histogram(prior_ensemble_ss[1,:], legend=false, title="β", xlim=(0, 1.0), bins=0:0.01:1.0)
p2 = histogram(prior_ensemble_ss[2,:], legend=false, title="i₀", xlim=(0, 0.1), bins=0:0.001:0.1)
p3 = histogram(post_samples_ss[1,:], legend=false, title="β", xlim=(0, 1.0), bins=0:0.01:1.0)
p4 = histogram(post_samples_ss[2,:], legend=false, title="i₀", xlim=(0, 0.1), bins=0:0.001:0.1)
plot(p1, p3, p2, p4, layout=l)


N_iterations_cs = 50
N_ensemble_cs = 10000
rng_cs = Random.Xoshiro(seed)
initial_ensemble_cs = EKP.construct_initial_ensemble(rng_cs, prior, N_ensemble_cs);


eks_obj_cases = EKP.EnsembleKalmanProcess(initial_ensemble_cs, cases, Γ, eks_process; rng = rng_cs)
for i in 1:N_iterations_cs
    params_i = get_ϕ_final(prior, eks_obj_cases)
    # cs = hcat([get_cases(params_i[:, i]) for i in 1:N_ensemble_cs]...)
    cs = hcat(ThreadsX.collect(get_cases(params_i[:, i]) for i in 1:N_ensemble_cs)...)
    EKP.update_ensemble!(eks_obj_cases, cs)
end


θ_post_cases = get_u_mean_final(eks_obj_cases)
Γ_post_cases = get_u_cov_final(eks_obj_cases);


post_samples_cases_unconstrained = rand(MvNormal(θ_post_cases, Γ_post_cases), 10000)
post_samples_cases = transform_unconstrained_to_constrained(prior, post_samples_cases_unconstrained);


quantile(post_samples_cases[1,:], [0.025, 0.5, 0.975])


quantile(post_samples_cases[2,:], [0.025, 0.5, 0.975])


corrplot(post_samples_cases',labels=["β" "i₀"])


prior_ensemble_cases = get_ϕ(prior, eks_obj_cases, 1);


l = @layout [a b; c d]
p1 = histogram(prior_ensemble_cases[1,:], legend=false, title="β", xlim=(0, 1.0), bins=0:0.01:1.0)
p2 = histogram(prior_ensemble_cases[2,:], legend=false, title="i₀", xlim=(0, 0.1), bins=0:0.001:0.1)
p3 = histogram(post_samples_cases[1,:], legend=false, title="β", xlim=(0, 1.0), bins=0:0.01:1.0)
p4 = histogram(post_samples_cases[2,:], legend=false, title="i₀", xlim=(0, 0.1), bins=0:0.001:0.1)
plot(p1, p3, p2, p4, layout=l)

