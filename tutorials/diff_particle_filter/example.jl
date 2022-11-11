# activate tutorial project file

# load dependencies
using StochasticAD
using Distributions
using DistributionsAD
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Zygote
using ForwardDiff
using GaussianDistributions
using GaussianDistributions: correct, ⊕
using Measurements
using UnPack
using Plots
using LaTeXStrings
using BenchmarkTools

struct StochasticModel{TType<:Integer,T1,T2,T3}
    T::TType # time steps
    start::T1 # prior
    dyn::T2 # dynamical model
    obs::T3 # observation model
end

struct ParticleFilter{mType<:Integer,MType<:StochasticModel,yType,sType}
    m::mType # number of particles
    StochM::MType # stochastic model
    ys::yType # observations
    sample_strategy::sType # sampling function
end

llikelihood(yres, S) = GaussianDistributions.logpdf(Gaussian(zero(yres), Symmetric(S)), yres)
struct KalmanFilter{dType<:Integer,MType<:StochasticModel,HType,RType,QType,yType}
    # H, R = obs
    # θ, Q = dyn
    d::dType
    StochM::MType # stochastic model
    H::HType # observation model, maps the true state space into the observed space
    R::RType # observation model, covariance matrix
    Q::QType # dynamical model, covariance matrix
    ys::yType # observations
end

function simulate_single(StochM::StochasticModel, θ)
    @unpack T, start, dyn, obs = StochM
    x = rand(start(θ))
    y = rand(obs(x, θ))
    xs = [x]
    ys = [y]
    for t in 2:T
        x = rand(dyn(x, θ))
        y = rand(obs(x, θ))
        push!(xs, x)
        push!(ys, y)
    end
    xs, ys
end

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

function resample(m, X, W, ω, sample_strategy, use_new_weight=true)
    js = Zygote.ignore(() -> sample_strategy(W, m, ω))
    X_new = X[js]
    if use_new_weight
        # differentiable resampling
        W_chosen = W[js]
        W_new = map(w -> ω * new_weight(w / ω) / m, W_chosen)
    else
        # stop gradient, biased approach
        W_new = fill(ω / m, m)
    end
    X_new, W_new
end

function (F::ParticleFilter)(θ; store_path=false, use_new_weight=true, s=1)
    # s controls the number of resampling steps
    @unpack m, StochM, ys, sample_strategy = F
    @unpack T, start, dyn, obs = StochM


    X = [rand(start(θ)) for j in 1:m] # particles
    W = [1 / m for i in 1:m] # weights
    ω = 1 # total weight
    store_path && (Xs = [X])
    for (t, y) in zip(1:T, ys)
        # update weights & likelihood using observations
        wi = map(x -> pdf(obs(x, θ), y), X)
        W = W .* wi
        ω_old = ω
        ω = sum(W)
        # resample particles
        if t > 1 && t < T && (t % s == 0) # && 1 / sum((W / ω) .^ 2) < length(W) ÷ 32
            X, W = resample(m, X, W, ω, sample_strategy, use_new_weight)
        end
        # update particle states
        if t < T
            X = map(x -> rand(dyn(x, θ)), X)
            store_path && Zygote.ignore(() -> push!(Xs, X))
        end
    end
    (store_path ? Xs : X), W
end

function (F::KalmanFilter)(θ)
    @unpack d, StochM, H, R, Q = F
    @unpack start = StochM

    x = start(θ)
    Φ = reshape(θ, d, d)

    x, yres, S = GaussianDistributions.correct(x, ys[1] + R, H)
    ll = llikelihood(yres, S)
    xs = Any[x]
    for i in 2:length(ys)
        x = Φ * x ⊕ Q
        x, yres, S = GaussianDistributions.correct(x, ys[i] + R, H)
        ll += llikelihood(yres, S)

        push!(xs, x)
    end
    xs, ll
end

function log_likelihood(F::ParticleFilter, θ, use_new_weight=true, s=1)
    _, W = F(θ; store_path=false, use_new_weight=use_new_weight, s=s)
    log(sum(W))
end

function log_likelihood(F::KalmanFilter, θ)
    _, ll = F(θ)
    ll
end

forw_grad(θ, F::ParticleFilter; s=1) = ForwardDiff.gradient(θ -> log_likelihood(F, θ, true, s), θ)
back_grad(θ, F::ParticleFilter; s=1) = Zygote.gradient(θ -> log_likelihood(F, θ, true, s), θ)[1]
forw_grad_biased(θ, F::ParticleFilter; s=1) = ForwardDiff.gradient(θ -> log_likelihood(F, θ, false, s), θ)
forw_grad_Kalman(θ, F::KalmanFilter) 

seed = 423897

### Define model
# here: n-dimensional rotation matrix
Random.seed!(seed)
T = 20 # time steps
d = 2 # dimension
# generate a rotation matrix
M = randn(d, d)
c = 0.3 # scaling
O = exp(c * (M - transpose(M)) / 2)
@assert det(O) ≈ 1
@assert transpose(O) * O ≈ I(d)
θtrue = vec(O) # true parameter

# observation model
R = 0.01 * collect(I(d))
obs(x, θ) = MvNormal(x, R) # y = H x + ν with ν ~ Normal(0, R)

# dynamical model
Q = 0.02 * collect(I(d))
dyn(x, θ) = MvNormal(reshape(θ, d, d) * x, Q) #  x = Φ*x + w with w ~ Normal(0,Q)

# starting position
x0 = randn(d)
# prior distribution
start(θ) = Gaussian(x0, 0.001 * collect(I(d)))

# put it all together
stochastic_model = StochasticModel(T, start, dyn, obs)

# relevant corresponding Kalman filterng defs
H_Kalman = collect(I(d))
R_Kalman = Gaussian(zeros(Float64, d), R)
# Φ_Kalman = O
Q_Kalman = Gaussian(zeros(Float64, d), Q)
###

### simulate model
Random.seed!(seed)
xs, ys = simulate_single(stochastic_model, θtrue)

m = 1000
kalman_filter = KalmanFilter(d, stochastic_model, H_Kalman, R_Kalman, Q_Kalman, ys)
particle_filter = ParticleFilter(m, stochastic_model, ys, sample_stratified)

### run and visualize filters
Xs, W = particle_filter(θtrue; store_path=true)
fig = plot(getindex.(xs, 1), getindex.(xs, 2), legend=false, xlabel=L"x_1", ylabel=L"x_2") # x1 and x2 are bad names..conflicting notation
scatter!(fig, getindex.(ys, 1), getindex.(ys, 2))
for i in 1:min(m, 100) # note that Xs has obs noise.
    local xs = [Xs[t][i] for t in 1:T]
    scatter!(fig, getindex.(xs, 1), getindex.(xs, 2), marker_z=1:T, color=:cool, alpha=0.1) # color to indicate time step
end

xs_Kalman, ll_Kalman = kalman_filter(θtrue)
plot!(getindex.(mean.(xs_Kalman), 1), getindex.(mean.(xs_Kalman), 2), legend=false, color="red")




### compute gradients
Random.seed!(seed)
X = [forw_grad(θtrue, particle_filter) for i in 1:200] # gradient of the particle filter *with* differentiation of the resampling step
Random.seed!(seed)
Xbiased = [forw_grad_biased(θtrue, particle_filter) for i in 1:200] # Gradient of the particle filter *without* differentiation of the resampling step
# pick an arbitrary coordinate
index = 1 # take derivative with respect to first parameter (2-dimensional example has a rotation matrix with four parameters in total)
# plot histograms for the sampled derivative values
fig = plot(normalize(fit(Histogram, getindex.(X, index), nbins=20), mode=:pdf), legend=false) # ours
plot!(normalize(fit(Histogram, getindex.(Xbiased, index), nbins=20), mode=:pdf)) # biased
vline!([mean(X)[index]], color=1)
vline!([mean(Xbiased)[index]], color=2)
# add derivative of differentiable Kalman filter as a comparison
XK = forw_grad_Kalman(θtrue, kalman_filter)
vline!([XK[index]], color="black")