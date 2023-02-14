using MeasureTheory
import Distributions
using Plots
using Random

# mc = Chain(x -> Normal(μ=x), Normal(μ=0.0))
mc = Chain(x -> Normal(μ=x), Dirac(0.0))
r = rand(mc)
samp = Iterators.take(r, 1000)
plot(collect(samp))


function counting(x)
    # [Dirac(x[1]+1), Dirac(x[2]-1)]
    Dirac([x[1]+1, x[2]-1])
end

# mc = Chain(x -> counting(x), [Dirac(1),Dirac(2)])
mc = Chain(x -> counting(x), Dirac([0,0]))
r = rand(mc)
samp = Iterators.take(r, 1000)
plot(transpose(hcat(collect(samp)...)))

# SIR

# needed until https://github.com/cscherrer/MeasureTheory.jl/issues/248 is looked at
function Base.rand(
    rng::AbstractRNG,
    ::Type,
    d::Binomial{(:n, :p),Tuple{I,A}},
) where {I<:Integer,A}
    rand(rng.rng, Dists.Binomial(d.n, d.p))
end

rate2prob(x) = Distributions.cdf(Distributions.Exponential(), x)

state = [990,10,0]
pars = (0.01, 0.25, 0.1)

# x is the counting processes N(t)
# state is the state of the system
function sir_trans(x, state, pars)
    β, γ, Δt = pars
    state[1] -= x[1]
    state[2] += x[1]
    state[2] -= x[2]
    state[3] += x[2]
    
    S, I, R = state
    # S -= x[1]
    # I += x[1]
    # I -= x[2]
    # R += x[2]

    si_prob = rate2prob(I*β*Δt)
    ir_prob = rate2prob(γ*Δt)
    si_rv = MeasureTheory.Binomial(S, si_prob)
    ir_rv = MeasureTheory.Binomial(I, ir_prob)
    
    # can probably use a product measure, since they are independent draws
    productmeasure([si_rv, ir_rv])
end

# e.g. this seems to produce reasonable stuff
# productmeasure([Dirac(5),Binomial(5,0.5)])

mc = Chain(x -> sir_trans(x, state, pars), productmeasure([Dirac(0), Dirac(0)]))
# r = rand(mc)
r = rand(Random.Xoshiro(), mc)
samp = Iterators.take(r, 1000)
sir_trace = collect(samp)
sir_trace = transpose(hcat(sir_trace...))

sir_out = zeros(Int,1001,3)
sir_out[1,:] = [990, 10, 0]
for i in 2:1001
    tmp_state = deepcopy(sir_out[i-1,:])
    tmp_state[1] -= sir_trace[i-1,1]
    tmp_state[2] += sir_trace[i-1,1]
    tmp_state[2] -= sir_trace[i-1,2]
    tmp_state[3] += sir_trace[i-1,2]
    sir_out[i,:] = tmp_state
end

plot(sir_out)

# check if this is general problem of product distn
using Random
using MeasureTheory
rng = ResettableRNG(Random.Xoshiro(), 6542022242862247233)
mu = ProductMeasure([Binomial(n = 990, p = 0.00995017), Binomial(n = 10, p = 0.0246901)])
rand(rng, mu)
rand(mu)