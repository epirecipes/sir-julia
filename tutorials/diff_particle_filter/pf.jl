using StochasticAD
using Distributions
using DistributionsAD
using Zygote
using ForwardDiff
using StaticArrays

using Plots
using Statistics

# ----------------------------------------------------------------------
# simulation code
# ----------------------------------------------------------------------

@inline function rate_to_proportion(r, t)
    1-exp(-r*t)
end

function dyn(x, p, Δt)
    S,I,R = x
    β,γ = p
    N = S+I+R
    ifrac = rate_to_proportion(β*I/N,Δt)
    rfrac = rate_to_proportion(γ,Δt)
    infection = rand(Binomial(S,ifrac))
    recovery = rand(Binomial(I,rfrac))
    return [S - infection, I + infection - recovery, R + recovery]
end

function simulate_single(nsteps::Integer, x0, p, Δt)
    xs = zeros(eltype(x0), nsteps, 3)
    x = copy(x0)
    xs[1, :] = x
    for n in 2:nsteps
        x = dyn(x, p, Δt)
        xs[n, :] = x
    end
    return xs
end

function simulate_multiple(nreps, nsteps, x0, p, Δt)
    reps = zeros(eltype(x0), nsteps, 3, nreps)
    for n in 1:nreps
        reps[:,:,n] = simulate_single(nsteps, x0, p, Δt)
    end
    return reps
end


# ----------------------------------------------------------------------
# plot sims to test
# ----------------------------------------------------------------------

const p = [0.5, 0.25]
const Δt = 0.1
const x0 = [990, 10, 0]

nsteps = 400
tmax = nsteps*Δt

trajs = simulate_multiple(100, nsteps, x0, p, Δt)

trajs_qt_S = hcat([quantile(trajs[i,1,:], [0.025, 0.5, 0.975]) for i=axes(trajs,1)]...)
trajs_qt_I = hcat([quantile(trajs[i,2,:], [0.025, 0.5, 0.975]) for i=axes(trajs,1)]...)
trajs_qt_R = hcat([quantile(trajs[i,3,:], [0.025, 0.5, 0.975]) for i=axes(trajs,1)]...)

plot(trajs_qt_S[2,:], color = 1)
plot!(trajs_qt_S[3,:], fillrange = trajs_qt_S[1,:], alpha = 0.25, color = 1)

plot!(trajs_qt_I[2,:], color = 2)
plot!(trajs_qt_I[3,:], fillrange = trajs_qt_I[1,:], alpha = 0.25, color = 2)

plot!(trajs_qt_R[2,:], color = 3)
plot!(trajs_qt_R[3,:], fillrange = trajs_qt_R[1,:], alpha = 0.25, color = 3, legend = false)


# ----------------------------------------------------------------------
# fake data for particle filter
# ----------------------------------------------------------------------

traj = simulate_single(nsteps, x0, p, Δt)
data = collect(10:10:400) # times at which we observe data
data = hcat(data, rand.(Poisson.(traj[data, 2])))


# ----------------------------------------------------------------------
# particle filter
# ----------------------------------------------------------------------

# resampling particles
function sample_stratified(p, K, sump=1)
    n = length(p)
    U = rand()
    is = zeros(Int, K)
    i = 1
    cw = p[1]
    for k in 1:K
        t = sump * (k - 1 + U) / K
        while cw < t && i < n
            i += 1
            @inbounds cw += p[i]
        end
        is[k] = i
    end
    return is
end

function resample(m, X, W, ω, sample_strategy)
    js = Zygote.ignore(() -> sample_strategy(W, m, ω))
    X_new = X[js]
    # differentiable resampling
    W_chosen = W[js]
    W_new = map(w -> ω * new_weight(w / ω) / m, W_chosen)
    X_new, W_new
end

function particle_filter(data, m, x0, p, Δt; store_path = false)
    X = fill(x0, m)
    W = [1/m for i in 1:m]
    ω = 1.0 # total weight
    
    n = 1 # timestep of particles
    store_path && (Xs = [X])
    
    for i in axes(data,1)
        t = data[i,1]
        # propagate particles to next data time point
        while n < t
            X = map(x -> dyn(x,p,Δt), X)
            n += 1
        end
        store_path && Zygote.ignore(() -> push!(Xs, X))
        # update weights
        wi = map((x) -> pdf(Poisson(x[2]), data[i,2]), X)
        W = W .* wi
        ω = sum(W)
        # resample particles
        if i < size(data,1)
            X, W = resample(m, X, W, ω, sample_stratified)
        end
    end
    
    return (store_path ? Xs : X), W
end

# get the estimated (filtered) log-likelihood
function log_likelihood(p, Δt, data, m, x0)
    _, W = particle_filter(data, m, x0, p, Δt, store_path=false)
    log(sum(W))
end


# ----------------------------------------------------------------------
# visualize the particle filter
# ----------------------------------------------------------------------

pf_out, pf_W = particle_filter(data, 100, x0, p, Δt, store_path = true)

plot(data[:,2], legend = false)
for i in 1:40
    scatter!(fill(i, 100), map((x) -> x[2], pf_out[i]), color=:red, alpha=0.05, legend = false)
end


# ----------------------------------------------------------------------
# differentiate ll wrt parameters
# ----------------------------------------------------------------------

ForwardDiff.gradient(p -> log_likelihood(p, Δt, data, 100, x0), p)
# Zygote.gradient(p -> log_likelihood(p, Δt, data, 100, x0), p)