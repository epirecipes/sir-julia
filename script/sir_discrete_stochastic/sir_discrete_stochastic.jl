
using DifferentialEquations
using SimpleDiffEq
using Distributions
using Random
using Plots
using BenchmarkTools


@inline function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end


function sir_discrete_stochastic(du,u,p,t)
    (S,I,R) = u
    (β,γ,δt) = p
    N = S+I+R
    ifrac = rate_to_proportion(β*I/N,δt)
    rfrac = rate_to_proportion(γ,δt)
    infection=rand(Binomial(S,ifrac))
    recovery=rand(Binomial(I,rfrac))
    @inbounds begin
        du[1] = S-infection
        du[2] = I+infection-recovery
        du[3] = R+recovery
    end
    nothing
end


δt = 0.01
nsteps = 5000
tf = nsteps*δt
tspan = (0.0,nsteps)


u0 = [999,1,0]
p = [0.5,0.25,0.01]


Random.seed!(1234)


prob_sir_discrete_stochastic = DiscreteProblem(sir_discrete_stochastic,u0,tspan,p)
sol_sir_discrete_stochastic = solve(prob_sir_discrete_stochastic,solver=FunctionMap)


plot(sol_sir_discrete_stochastic)


@benchmark solve(prob_sir_discrete_stochastic,solver=FunctionMap)

