using MeasureTheory
import Distributions
using Plots

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


rate2prob(x) = Distributions.cdf(Distributions.Exponential(), x)

function sir_trans(x, pars)
    β, γ, Δt = pars
    S, I, R = x
    si_prob = rate2prob(I*β*Δt)
    ir_prob = rate2prob(γ*Δt)
    si_rv = Binomial(S, si_prob)
    ir_rv = Binomial(I, ir_prob)
    [x[1]-si_rv, x[2]+si_rv-ir_rv, x[3]+ir_rv]
end

mc = Chain(x -> sir_trans, Dirac([990, 10, 0]))
r = rand(mc)
samp = Iterators.take(r, 1000)

# breakpoint(sir_trans)

state = [990,10,0]
pars = (0.01, 0.25, 0.1)

# x is the counting processes N(t)
# state is the state of the system
function sir_trans(x, state, pars)
    β, γ, Δt = pars
    S, I, R = state
    S -= x[1]
    I += x[1]
    I -= x[2]
    R += x[2]

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
r = rand(mc)
samp = Iterators.take(r, 1000)
collect(samp)


# check if this is general problem of product distn
using Random
using MeasureTheory
rng = ResettableRNG(Random.Xoshiro(), 6542022242862247233)
mu = ProductMeasure([Binomial(n = 990, p = 0.00995017), Binomial(n = 10, p = 0.0246901)])
rand(rng, mu)
rand(mu)