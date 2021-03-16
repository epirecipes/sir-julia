
using Random
using Soss
using DataFrames
using StatsPlots
using BenchmarkTools


@inline function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;


sir_markov = @model state,p begin
    # Unpack parameters
    β = p.β
    c = p.c
    γ = p.γ
    δt = p.δt

    # Unpack starting counts
    t0 = state.t
    S0 = state.S
    I0 = state.I
    R0 = state.R
    N = S0 + I0 + R0

    # Transitions between states
    S_I ~ Binomial(S0, rate_to_proportion(β*c*I0/N,δt))
    I_R ~ Binomial(I0, rate_to_proportion(γ,δt))

    # Updated state
    t = t0 + δt
    S = S0 - S_I
    I = I0 + S_I - I_R
    R = R0 + I_R

    next = (p=p, state=(t=t,S=S,I=I,R=R))
end;


sir_model = @model u0,p begin
    x ~ MarkovChain(p, sir_markov(state=u0,p=p))
end;


δt = 0.1
nsteps = 400
tmax = nsteps*δt;


u0 = (t=0.0, S=990, I=10, R=0); # t,S,I,R


p = (β=0.05, c=10.0, γ=0.25, δt=δt);


Random.seed!(1234);


r = rand(sir_model(u0=u0,p=p));
data = [u0]
for (n,s) in enumerate(r.x)
    n>nsteps && break
    push!(data,s)
end;


df_markov = DataFrame(data);


@df df_markov plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark begin
    r = rand(sir_model(u0=u0,p=p));
    data = [u0]
    for (n,s) in enumerate(r.x)
        n>nsteps && break
        push!(data,s)
    end
end

