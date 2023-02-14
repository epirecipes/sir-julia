using MeasureTheory
import Distributions
using Plots
using Random

pars = (β = 0.05*10.0, γ = 1/4, Δt = 0.01)
tmax = 40.0
nsteps = Int64(tmax / pars.Δt)
state = [990,10,0]

# needed until https://github.com/cscherrer/MeasureTheory.jl/issues/248 is looked at
function Base.rand(
    rng::AbstractRNG,
    ::Type,
    d::Binomial{(:n, :p),Tuple{I,A}},
) where {I<:Integer,A}
    rand(rng.rng, Dists.Binomial(d.n, d.p))
end

rate2prob(x) = Distributions.cdf(Distributions.Exponential(), x)

# x is the counting processes N(t)
# state is the state of the system
function sir_trans(x, state, pars)
    β, γ, Δt = pars
    state[1] -= x[1]
    state[2] += x[1]
    state[2] -= x[2]
    state[3] += x[2]

    S, I, R = state
    N = sum(state)

    si_prob = rate2prob(I/N*β*Δt)
    ir_prob = rate2prob(γ*Δt)
    si_rv = MeasureTheory.Binomial(S, si_prob)
    ir_rv = MeasureTheory.Binomial(I, ir_prob)
    
    # can probably use a product measure, since they are independent draws
    productmeasure([si_rv, ir_rv])
end

mc = Chain(x -> sir_trans(x, state, pars), productmeasure([Dirac(0), Dirac(0)]))
r = rand(Random.Xoshiro(), mc)
samp = Iterators.take(r, nsteps)
sir_trace = collect(samp)
sir_trace = transpose(hcat(sir_trace...))

sir_out = zeros(Int,nsteps+1,3)
sir_out[1,:] = [990, 10, 0]
for i in 2:nsteps+1
    tmp_state = deepcopy(sir_out[i-1,:])
    tmp_state[1] -= sir_trace[i-1,1]
    tmp_state[2] += sir_trace[i-1,1]
    tmp_state[2] -= sir_trace[i-1,2]
    tmp_state[3] += sir_trace[i-1,2]
    sir_out[i,:] = tmp_state
end

plot((0:nsteps)*pars.Δt, sir_out)
