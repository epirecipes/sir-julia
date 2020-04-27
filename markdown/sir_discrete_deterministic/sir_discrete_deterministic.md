
````julia
using DifferentialEquations
using SimpleDiffEq
using Plots
using BenchmarkTools
````



````julia
@inline function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end
````


````
rate_to_proportion (generic function with 1 method)
````



````julia
function sir_discrete_deterministic(du,u,p,t)
    (S,I,R) = u
    (β,γ,δt) = p
    N = S+I+R
    infection = rate_to_proportion(β*I/N,δt)*S
    recovery = rate_to_proportion(γ,δt)*I
    @inbounds begin
        du[1] = S-infection
        du[2] = I+infection-recovery
        du[3] = R+recovery
    end
    nothing
end
````


````
sir_discrete_deterministic (generic function with 1 method)
````



````julia
δt = 0.01
nsteps = 5000
tf = nsteps*δt
tspan = (0.0,nsteps)
````


````
(0.0, 5000)
````



````julia
u0 = [999,1,0]
p = [0.5,0.25,0.01]
prob_sir_discrete_deterministic = DiscreteProblem(sir_discrete_deterministic,u0,tspan,p)
sol_sir_discrete_deterministic = solve(prob_sir_discrete_deterministic,solver=FunctionMap)
````


````
Error: InexactError: Int64(998.9950050124875)
````



````julia
plot(sol_sir_discrete_deterministic)
````


````
Error: UndefVarError: sol_sir_discrete_deterministic not defined
````



````julia
@benchmark solve(prob_sir_discrete_deterministic,solver=FunctionMap)
````


````
Error: InexactError: Int64(998.9950050124875)
````


