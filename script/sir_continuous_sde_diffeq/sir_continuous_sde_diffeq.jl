
using DifferentialEquations
using StochasticDiffEq
using Random
using SparseArrays
using Plots


function sir_ode(du,u,p,t)
    (S,I,R) = u
    (β,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*S*I/N
        du[2] = β*S*I/N - γ*I
        du[3] = γ*I
    end
    nothing
end


# Define a sparse matrix by making a dense matrix and setting some values as not zero
A = zeros(3,2)
A[1,1] = 1
A[2,1] = 1
A[2,2] = 1
A[3,2] = 1
A = SparseArrays.sparse(A)


# Make `g` write the sparse matrix values
function sir_noise(du,u,p,t)
    (S,I,R) = u
    (β,γ) = p
    N = S+I+R
    ifrac = β*I/N*S
    rfrac = γ*I
    du[1,1] = -sqrt(ifrac)
    du[2,1] = sqrt(ifrac)
    du[2,2] = -sqrt(rfrac)
    du[3,2] = sqrt(rfrac)
end


tspan = (0.0,50.0)
u0 = [999.0,1.0,0.0]
p = [0.5,0.25]
Random.seed!(1234)


prob_sir_sde = SDEProblem(sir_ode,sir_noise,u0,tspan,p,noise_rate_prototype=A)
sol_sir_sde = solve(prob_sir_sde,SRA1())


plot(sol_sir_sde,vars=[(0,1),(0,2),(0,3)])

