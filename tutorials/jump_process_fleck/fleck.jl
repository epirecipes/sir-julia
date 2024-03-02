using Fleck
using Distributions
using Random
using Plots

u0 = [990, 10, 0]; # S, I, R
p = [0.05, 10.0, 0.25]; # β, c, γ
tmax = 40.0

# a little struct to hold the model
mutable struct SIRNonMarkov{T<:Distribution}
    state::Vector{Int}
    parameters::Vector{Float64}
    next_key::Int
    infection_rate::Float64
    recovery_distribution::T
    time::Float64
end

function get_key!(model::SIRNonMarkov)
    key = model.next_key
    model.next_key += 1
    return key
end

function initialize!(model::SIRNonMarkov, sampler, rng)
    (β, c, γ) = model.parameters
    # enable the infection transition
    enable!(sampler, (:infection, get_key!(model)), Exponential(1.0/(β*c*model.state[2]/sum(model.state)*model.state[1])), model.time, model.time, rng)
    # enable the recovery transitions
    for _ in 1:model.state[2]
        enable!(sampler, (:recovery, get_key!(model)), model.recovery_distribution, model.time, model.time, rng)
    end
end

function step!(model::SIRNonMarkov, sampler, when, which, rng)
    (β, c, γ) = model.parameters
    model.time = when
    if first(which) == :infection
        model.state[1] -= 1
        model.state[2] += 1
        # disable and reenable the infection clock after accounting for the new rate
        disable!(sampler, which, model.time)
        enable!(sampler, which, Exponential(1.0/(β*c*model.state[2]/sum(model.state)*model.state[1])), model.time, model.time, rng)
        # enable a recovery event for the newly infected person
        enable!(sampler, (:recovery, get_key!(model)), model.recovery_distribution, model.time, model.time, rng)
    elseif first(which) == :recovery
        model.state[2] -= 1
        model.state[3] += 1
        disable!(sampler, which, model.time)
    else
        error("unrecognized clock key: $(which)")
    end
end

sirmodel = SIRNonMarkov(deepcopy(u0), p, 0, prod(p[1:2]), Dirac(1/p[3]), 0.0)
sampler = CombinedNextReaction{Tuple{Symbol,Int}}()
rng = Xoshiro()

output = zeros(prod(sirmodel.state[1:2])+sum(sirmodel.state[1:2])+1, 4)
nout = 1
output[nout,:] = [sirmodel.time; sirmodel.state]
nout += 1

initialize!(sirmodel, sampler, rng)

(when, which) = next(sampler, sirmodel.time, rng)

while when ≤ tmax
    step!(sirmodel, sampler, when, which, rng)
    (when, which) = next(sampler, sirmodel.time, rng)

    output[nout,:] = [sirmodel.time; sirmodel.state]
    nout += 1
end

plot(output[1:nout-1,1], output[1:nout-1,2:end])