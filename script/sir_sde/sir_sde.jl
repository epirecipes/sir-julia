
using DifferentialEquations
using SimpleDiffEq
using Distributions
using Random
using Plots
using BenchmarkTools


function sir_sde(du,u,p,t)
    (S,I,R) = u
    (β,γ,δt) = p
    N = S+I+R
    ifrac = β*I/N*S*δt
    rfrac = γ*I*δt
    ifrac_noise = sqrt(ifrac)*rand(Normal(0,1))
    rfrac_noise = sqrt(rfrac)*rand(Normal(0,1))
    @inbounds begin
        du[1] = S-(ifrac+ifrac_noise)
        du[2] = I+(ifrac+ifrac_noise) - (rfrac + rfrac_noise)
        du[3] = R+(rfrac+rfrac_noise)
    end
    for i in 1:3
        if du[i] < 0 du[i]=0 end
    end
    nothing
end


δt = 0.01
nsteps = 5000
tf = nsteps*δt
tspan = (0.0,nsteps)


u0 = [990.0,10.0,0.0]
p = [0.5,0.25,0.01]


Random.seed!(1234)


prob_sir_sde = DiscreteProblem((du,u,p,t)->sir_sde(du,u,p,t),u0,tspan,p)
sol_sir_sde = solve(prob_sir_sde,solver=FunctionMap)


plot(sol_sir_sde)


@benchmark solve(prob_sir_sde,solver=FunctionMap)

