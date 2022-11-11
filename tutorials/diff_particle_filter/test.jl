using StochasticAD
using Distributions
using DistributionsAD
using Zygote
using ForwardDiff
using StaticArrays

using Plots
using Statistics

@inline function rate_to_proportion(r, t)
    1-exp(-r*t)
end

# struct StochasticModel{TType<:Integer,T1,T2,T3}
#     T::TType # time steps
#     start::T1 # prior
#     dyn::T2 # dynamical model
#     obs::T3 # observation model
# end

# struct SIR
#     dyn::T1
#     obs::T2
# end

# struct SIRparticle{T <: Integer}
#     x::SVector{3, T}
# end

function dyn(x::T, θ) where {T <: AbstractVector}
    S,I,R = x
    (β,γ) = θ
    N = S+I+R
    ifrac = rate_to_proportion(β*I/N,Δt)
    rfrac = rate_to_proportion(γ,Δt)
    infection=rand(Binomial(S,ifrac))
    recovery=rand(Binomial(I,rfrac))
    return T(S - infection, I + infection - recovery, R + recovery)
end



# x = SVector{3, Int64}(50,20,5)
# θ = [0.5, 0.25, 0.1]

# dyn(x, θ)
# @code_warntype dyn(x, θ)

function simulate_single(nsteps::Integer, x0::T, θ) where {T <: AbstractVector}
    xs = zeros(eltype(x0), nsteps, 3)
    x = T(x0)
    xs[1, :] = x
    for n in 2:nsteps
        x = dyn(x, θ)
        xs[n, :] = x
    end
    return xs
end


θ = [0.5, 0.25]
Δt = 0.1

nsteps = 400
tmax = nsteps*Δt
x0 = SVector{3, Int64}(990, 10, 0)

traj = simulate_single(nsteps, x0, θ)

# @code_warntype simulate_single(nsteps, x0, θ)

plot(traj)


function simulate_multiple(nreps::Integer, nsteps::Integer, x0::T, θ) where {T <: SVector}
    reps = zeros(eltype(x0), nsteps, 3, nreps)
    for n in 1:nreps
        reps[:,:,n] = simulate_single(nsteps, x0, θ)
    end
    return reps
end

trajs = simulate_multiple(100, nsteps, x0, θ)

trajs_qt_S = hcat([quantile(trajs[i,1,:], [0.025, 0.5, 0.975]) for i=axes(trajs,1)]...)
trajs_qt_I = hcat([quantile(trajs[i,2,:], [0.025, 0.5, 0.975]) for i=axes(trajs,1)]...)
trajs_qt_R = hcat([quantile(trajs[i,3,:], [0.025, 0.5, 0.975]) for i=axes(trajs,1)]...)

plot(trajs_qt_S[2,:], color = 1)
plot!(trajs_qt_S[3,:], fillrange = trajs_qt_S[1,:], alpha = 0.25, color = 1)

plot!(trajs_qt_I[2,:], color = 2)
plot!(trajs_qt_I[3,:], fillrange = trajs_qt_I[1,:], alpha = 0.25, color = 2)

plot!(trajs_qt_R[2,:], color = 3)
plot!(trajs_qt_R[3,:], fillrange = trajs_qt_R[1,:], alpha = 0.25, color = 3, legend = false)


# make some data
data = collect(10:10:400) # times at which we observe data
data = hcat(data, rand.(Poisson.(traj[data, 2])))



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



# # the particle filter
# m = 100 # number of particles

# X = fill(x0, m)
# W = [1/m for i in 1:m]
# ω = 1.0 # total weight

# n = 1 # timestep of particles

# for i in axes(data,1)
#     t = data[i,1]
#     # propagate particles to next data time point
#     while n < t
#         # update all particles
#         for j in 1:m
#             X[j] = dyn(X[j],θ)
#         end
#         n += 1
#     end
#     # update weights
#     # wi = map((x) -> pdf(Poisson(x[2]), data[i,2]), X)
#     # W = W .* wi
#     # ω = sum(W)
#     W = map((x) -> pdf(Poisson(x[2]), data[i,2]), X)
#     ω = sum(W)

#     # resample particles
#     if t < size(data,1)
#         X, W = resample(m, X, W, ω, sample_strategy)
#     end
# end


function particle_filter(data, m::Integer, x0::T, θ; store_path = false) where {T <: SVector}
    X = fill(x0, m)
    W = [1/m for i in 1:m]
    ω = 1.0 # total weight
    
    n = 1 # timestep of particles

    store_path && (Xs = [X])
    
    for i in axes(data,1)
        t = data[i,1]
        # propagate particles to next data time point
        while n < t
            # update all particles
            # for j in 1:m
            #     X[j] = dyn(X[j],θ)
            # end
            X = map(x -> dyn(x,θ), X)
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


Xs, W = particle_filter(data, 1000, x0, θ, store_path=true)

log(sum(W)) # estimated log-likelihood

# plot filtered trajectories
m = 1000
plot(data[:,1], data[:,2])

filter_I = Float64[]
filter_t = Float64[]
for t in 1:size(Xs,1)
    for j in 1:m  
        push!(filter_t, t == 1 ? 0 : t/θ[end] - 10) 
        push!(filter_I, Xs[t][j][2])
    end
end

scatter!(filter_t, filter_I, alpha = 0.05, legend = false)


# get the estimated (filtered) log-likelihood
function log_likelihood(θ, data, m, x0)
    _, W = particle_filter(data, m, x0, θ, store_path=false)
    log(sum(W))
end

m = 1000
ForwardDiff.gradient(θ -> log_likelihood(θ, data, m, x0), θ)
Zygote.gradient(θ -> log_likelihood(θ, data, m, x0), θ)



x0 = [990,10,0]

function epi_size(p)
    simulate_single(nsteps, x0, p)[end,3]
end

ForwardDiff.gradient(θ -> epi_size(θ), θ)

derivative_estimate(θ -> epi_size(θ), θ)

using Debugger

stochastic_triple(θ -> epi_size(θ), θ)

# issue: can't propogate the stochastic_Triple through the T() ctor in the dynamics function;
# need to find a way to code time stepping update functions which can return stochastic_triples

# code a single function which does this.

θ = [0.5, 0.25]
Δt = 0.1

nsteps = 400
tmax = nsteps*Δt
x0 = SVector{3, Int64}(990, 10, 0)


function dyn1(x::T, θ) where {T <: AbstractVector}
    S,I,R = x
    (β,γ) = θ
    N = S+I+R
    ifrac = rate_to_proportion(β*I/N,Δt)
    rfrac = rate_to_proportion(γ,Δt)
    infection=rand(Binomial(S,ifrac))
    recovery=rand(Binomial(I,rfrac))
    return [S - infection, I + infection - recovery, R + recovery]
end

function simulate_single1(nsteps::Integer, x0::T, θ) where {T <: AbstractVector}
    
    x = x0

    for n in 2:nsteps
        x = dyn1(x, θ)
    end
    return x
end

function epi_size1(p)
    simulate_single1(nsteps, x0, p)[3]
end

derivative_estimate(θ -> epi_size1(θ), θ)
stochastic_triple(θ -> epi_size1(θ), θ)
