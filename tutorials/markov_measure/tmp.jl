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

breakpoint(sir_trans)

