using StochasticAD
using Distributions
using DistributionsAD
using Zygote
using ForwardDiff
using StaticArrays

@inline function rate_to_proportion(r, t)
    1-exp(-r*t)
end

struct StochasticModel{TType<:Integer,T1,T2,T3}
    T::TType # time steps
    start::T1 # prior
    dyn::T2 # dynamical model
    obs::T3 # observation model
end

struct SIR
    dyn::T1
    obs::T2
end

struct SIRparticle{T <: Integer}
    x::SVector{3, T}
end

function dyn(x::T, θ) where {T <: SVector}
    S,I,R = x
    (β,γ,δt) = θ
    N = S+I+R
    ifrac = rate_to_proportion(β*I/N,δt)
    rfrac = rate_to_proportion(γ,δt)
    infection=rand(Binomial(S,ifrac))
    recovery=rand(Binomial(I,rfrac))
    return T(S - infection, I + infection - recovery, R + recovery)
end

x = SVector{3, Int64}(50,20,5)
θ = [1, 0.05, 1]

dyn(x, θ)
@code_warntype dyn(x, θ)