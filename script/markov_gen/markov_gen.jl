
using OrdinaryDiffEq
using SciMLSensitivity
using Distributions
using Random
using Gen
using GenDistributions # to define custom distributions
using GenParticleFilters # for SMC
using Plots;


Random.seed!(1234);


@gen function sir_markov_model(l::Int=40,
                               N::Int=1000,
                               c::Float64=10.0,
                               γ::Float64=0.25,
                               δt::Float64=1.0)
    i₀ ~ uniform_discrete(1, 100)
    β ~ uniform(0.01, 0.1)
    S = N - i₀
    I = i₀
    for i in 1:l
        ifrac = 1-exp(-β*c*I/N*δt)
        rfrac = 1-exp(-γ*δt)
        infection = {(:y, i)} ~ binom(S,ifrac)
        recovery = {(:z, i)} ~ binom(I,rfrac)
        S = S-infection
        I = I+infection-recovery
    end
    nothing
end;


p = Gen.choicemap()
p[:β] = 0.05
p[:i₀] = 10
fixed_args = (1000, 10.0, 0.25, 1.0)
l = 40
(sol, _) = Gen.generate(sir_markov_model, (l,fixed_args...), p);


Gen.get_args(sol)


sol[:β]


sol[:i₀]


ts = collect(range(1,l))
Y = [sol[(:y, i)] for i=1:l]
scatter(ts,Y,xlabel="Time",ylabel="Number",label=false)


observations = Gen.choicemap()
for (i, y) in enumerate(Y)
    observations[(:y, i)] = y
end;


num_particles = 1000
num_replicates = 1000
β_is = Vector{Real}(undef, num_replicates)
i₀_is = Vector{Int}(undef, num_replicates)
for i in 1:num_replicates
    (trace, lml_est) = Gen.importance_resampling(sir_markov_model, (l,fixed_args...), observations, num_particles)
    β_is[i] = trace[:β]
    i₀_is[i] = trace[:i₀]
end;


mean(β_is),sqrt(var(β_is))


mean(i₀_is),sqrt(var(i₀_is))


pl_β_is = histogram(β_is, label=false, title="β", ylabel="Density", density=true, xrotation=45)
vline!([sol[:β]], label="True value")
pl_i₀_is = histogram(i₀_is, label=false, title="i₀", ylabel="Density", density=true, xrotation=45)
vline!([sol[:i₀]], label="True value")
plot(pl_β_is, pl_i₀_is, layout=(1,2), plot_title="Importance sampling")


const truncated_normal = DistributionsBacked((mu,std,lb,ub) -> Distributions.Truncated(Normal(mu, std), lb, ub),
                                             (true,true,false,false),
                                             true,
                                             Float64)


@gen function sir_proposal(current_trace)
    β ~ truncated_normal(current_trace[:β], 0.01, 0.0, Inf)
    i₀ ~ uniform_discrete(current_trace[:i₀]-1, current_trace[:i₀]+1)
end;


n_iter = 100000
β_mh = Vector{Real}(undef, n_iter)
i₀_mh = Vector{Int}(undef, n_iter)
scores = Vector{Float64}(undef, n_iter)
(tr,) = Gen.generate(sir_markov_model, (l,fixed_args...), merge(observations, p))
n_accept = 0
for i in 1:n_iter
    global (tr, did_accept) = Gen.mh(tr, sir_proposal, ()) # Gen.mh(tr, select(:β, :i₀)) for untargeted
    β_mh[i] = tr[:β]
    i₀_mh[i] = tr[:i₀]
    scores[i] = Gen.get_score(tr)
    if did_accept
        global n_accept += 1
    end
end;


acceptance_rate = n_accept/n_iter


pl_β_mh = histogram(β_mh, label=false, title="β", ylabel="Density", density=true, xrotation=45)
vline!([sol[:β]], label="True value")
pl_i₀_mh = histogram(i₀_mh, label=false, title="i₀", ylabel="Density", density=true, xrotation=45)
vline!([sol[:i₀]], label="True value")
plot(pl_β_mh, pl_i₀_mh, layout=(1,2), plot_title="Metropolis-Hastings")


kern(tr) = Gen.mh(tr, sir_proposal, ());


function particle_filter(observations, n_particles, ess_thresh=0.5)
    # Initialize particle filter with first observation
    n_obs = length(observations)
    obs_choices = [choicemap((:y, t) => observations[t]) for t=1:n_obs]
    state = pf_initialize(sir_markov_model, (1,fixed_args...), obs_choices[1], n_particles)
    # Iterate across timesteps
    for t=2:n_obs
        # Resample if the effective sample size is too low
        if effective_sample_size(state) < ess_thresh * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Rejuvenate particles
            pf_rejuvenate!(state, kern, ())
        end
        # Update filter state with new observation at timestep t
        # The following code explicitly allows for the number of timesteps to change
        # while keeping the other arguments fixed
        new_args = (t, fixed_args...)
        argdiffs = (UnknownChange(), map(x -> NoChange(), new_args)...)
        pf_update!(state, new_args, argdiffs, obs_choices[t])
    end
    return state
end;


n_particles = 10000
state = particle_filter(Y, n_particles);


effective_sample_size(state)


mean(state, :β), sqrt(var(state, :β))


mean(state, :i₀), sqrt(var(state, :i₀))


β_smc = getindex.(state.traces, :β)
i₀_smc = getindex.(state.traces, :i₀)
w = get_norm_weights(state);


pl_β_smc = histogram(β_smc, label=false, title="β", ylabel="Density", density=true, xrotation=45, weights=w)
vline!([sol[:β]], label="True value")
pl_i₀_smc = histogram(i₀_smc, label=false, title="i₀", ylabel="Density", density=true, xrotation=45, weights=w)
vline!([sol[:i₀]], label="True value")
plot(pl_β_smc, pl_i₀_smc, layout=(1,2), plot_title="Sequential Monte Carlo")

