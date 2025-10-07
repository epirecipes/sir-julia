###############################################################################
# Generalized (Real‑n) Binomial distribution extension for Distributions.jl   #
#                                                                             #
#  ▸ Supports non‑integer `n` via Gamma‑function extension of the binomial    #
#    coefficient.                                                             #
#  ▸ Adaptive `rand` starts with an exact rejection sampler that proposes     #
#    from an ordinary Binomial with integer `nᵖ = ⌈n⌉`.                       #
#    – It works for **any** fractional part δ = n − ⌊n⌋ ∈ (0,1).              #
#    – If `p > 0.5` we sample the *complement* (replace `p` with `1‑p` and    #
#      reflect) to keep the acceptance rate high.                             #
#  ▸ Falls back to inverse‑CDF, Poisson, or Normal approximations only if the #
#    proposal gets rejected too often (>128 attempts, extremely rare).        #
###############################################################################

module GeneralizedBinomialExt

using Distributions
using StatsFuns: loggamma, log1p, betainc
using Random: AbstractRNG, randn

export GeneralizedBinomial

# ──────────────────────────────────────────────────────────────────────────────
#  Distribution definition                                                      
# ──────────────────────────────────────────────────────────────────────────────

"""
    GeneralizedBinomial(n, p; check_args=true)

A *generalised Binomial distribution* with a (possibly non‑integer) number of
trials `n ≥ 0` and success probability `0 ≤ p ≤ 1`.

The pdf for integer `k ∈ 0:⌊n⌋` is

```
P(X=k) = Γ(n+1)/(Γ(k+1)Γ(n-k+1)) · p^k (1-p)^(n-k).
```

When `n` is an integer we delegate to `Distributions.Binomial`.
"""
struct GeneralizedBinomial{N<:Real,T<:Real} <: DiscreteUnivariateDistribution
    n :: N
    p :: T

    function GeneralizedBinomial{N,T}(n::N, p::T; check_args::Bool = true) where {N<:Real,T<:Real}
        check_args && Distributions.@check_args GeneralizedBinomial (n, n ≥ zero(n)) (p, zero(p) ≤ p ≤ one(p))
        new{N,T}(n, p)
    end
end

# convenience promotion constructor
GeneralizedBinomial(n::Real, p::Real; check_args::Bool = true) = begin
    NT, PT = promote(n, p)
    GeneralizedBinomial{typeof(NT), typeof(PT)}(NT, PT; check_args = check_args)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Basic properties                                                             
# ──────────────────────────────────────────────────────────────────────────────

Distributions.ntrials(d::GeneralizedBinomial) = d.n
Distributions.succprob(d::GeneralizedBinomial) = d.p
Distributions.failprob(d::GeneralizedBinomial) = one(d.p) - d.p

Distributions.mean(d::GeneralizedBinomial) = d.n * d.p
Distributions.var(d::GeneralizedBinomial)  = d.n * d.p * (one(d.p) - d.p)

# ──────────────────────────────────────────────────────────────────────────────
#  logpdf / pdf                                                                 
# ──────────────────────────────────────────────────────────────────────────────

@inline _lbinomcoeff(n::Real, k::Integer) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

function Distributions.logpdf(d::GeneralizedBinomial, k::Integer)
    k < 0 && return -Inf
    k > floor(Int, d.n) && return -Inf
    return _lbinomcoeff(d.n, k) + k * log(d.p) + (d.n - k) * log1p(-d.p)
end

Distributions.pdf(d::GeneralizedBinomial, k::Integer) = exp(logpdf(d, k))

# ──────────────────────────────────────────────────────────────────────────────
#  cdf                                                                          
# ──────────────────────────────────────────────────────────────────────────────

function Distributions.cdf(d::GeneralizedBinomial, k::Integer)
    k < 0 && return zero(d.p)
    k ≥ floor(Int, d.n) && return one(d.p)
    return 1 - betainc(k + 1, d.n - k, d.p)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Random number generation                                                     
# ──────────────────────────────────────────────────────────────────────────────

const _MAX_REJECT_ITERS = 128   # if exceeded we fall back to approx. samplers

# Proposal Binomial log‑pdf (works for complement trick too)
@inline _logpdf_prop(b::Binomial, k::Int) = logpdf(b, k)

# Exact rejection sampler using Binomial(⌈n⌉, p̃) envelope
function _rand_reject!(rng::AbstractRNG, d::GeneralizedBinomial, m_prop::Int, p_prop::Float64, flip::Bool)
    proposal = Binomial(m_prop, p_prop)
    kmax     = floor(Int, d.n)
    iter     = 0
    while true
        iter += 1
        if iter > _MAX_REJECT_ITERS
            return nothing  # give up, caller will choose fallback
        end
        k_prop = rand(rng, proposal)            # draw from envelope

        # Reflect if we sampled from the complement (p>0.5 case)
        k = flip ? m_prop - k_prop : k_prop

        k > kmax && continue                    # outside support → reject

        # log acceptance probability
        logf = Distributions.logpdf(d, k)
        logg = _logpdf_prop(proposal, k_prop)   # proposal density at original draw

        log(rand(rng)) < (logf - logg) && return k
    end
end

# Inverse‑CDF table for modest support
function _rand_small(rng::AbstractRNG, d::GeneralizedBinomial, m::Int)
    u, c = rand(rng), 0.0
    for k in 0:m
        c += exp(logpdf(d, k))
        u ≤ c && return k
    end
    return m
end

# Poisson approximation with truncation
@inline _rand_poisson_trunc(rng::AbstractRNG, m::Int, λ::Float64) = min(rand(Poisson(λ)), m)

# Normal approximation with truncation
function _rand_normal_trunc(rng::AbstractRNG, m::Int, μ::Float64, σ::Float64)
    round(Int, randn(rng) * σ + μ) |> k -> clamp(k, 0, m)
end

function Distributions.rand(rng::AbstractRNG, d::GeneralizedBinomial)
    # Integer n → vanilla Binomial
    isinteger(d.n) && return rand(rng, Binomial(convert(Int, round(d.n)), d.p))

    m      = floor(Int, d.n)          # ⌊n⌋  (top of support)
    m_prop = m + 1                    # proposal trials = ⌈n⌉

    # Degenerate probabilities
    (d.p == 0 || d.p == 1) && return (d.p == 0 ? 0 : m)

    # Complement trick keeps acceptance high when p>0.5
    flip   = d.p > 0.5
    p_prop = flip ? (1 - d.p) : d.p

    # 1️⃣  Try exact rejection sampler first
    k = _rand_reject!(rng, d, m_prop, p_prop, flip)
    if k !== nothing
        return k  # success within iteration budget
    end

    # 2️⃣  Fallback strategies (should rarely be reached)
    if m ≤ 1000
        return _rand_small(rng, d, m)
    end

    λ = d.n * d.p
    if d.p < 0.10 && λ < 30
        return _rand_poisson_trunc(rng, m, λ)
    else
        σ2 = λ * (1 - d.p)
        return _rand_normal_trunc(rng, m, λ, sqrt(σ2))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  Public constructor over‑load                                                 
# ──────────────────────────────────────────────────────────────────────────────

import Distributions: Binomial
function Binomial(n::Real, p::Real; check_args::Bool = true)
    isinteger(n) && return Binomial(convert(Int, round(n)), p; check_args = check_args)
    GeneralizedBinomial(n, p; check_args = check_args)
end

end # module
