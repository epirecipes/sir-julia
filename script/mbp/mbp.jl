
using DifferentialEquations
using StochasticDiffEq
using DiffEqCallbacks
using Random
using SparseArrays
using DataFrames
using StatsPlots
using BenchmarkTools


function sir_mbp!(du,u,p,t)
    (Ctilde,Rtilde) = u
    (β,c,γ,S₀,I₀,N) = p
    C = exp(Ctilde)-1.0
    R = exp(Rtilde)-1.0
    S = S₀-C
    I = I₀+C-R
    @inbounds begin
        du[1] = (exp(-Ctilde)-0.5*exp(-2.0*Ctilde))*(β*c*I/N*S)
        du[2] = (exp(-Rtilde)-0.5*exp(-2.0*Rtilde))*(γ*I)
    end
    nothing
end;


# Define a sparse matrix by making a dense matrix and setting some values as not zero
A = zeros(2,2)
A[1,1] = 1
A[2,2] = 1
A = SparseArrays.sparse(A);


# Make `g` write the sparse matrix values
function sir_noise!(du,u,p,t)
    (Ctilde,Rtilde) = u
    (β,c,γ,S₀,I₀,N) = p
    C = exp(Ctilde)-1.0
    R = exp(Rtilde)-1.0
    S = S₀-C
    I = I₀+C-R
    du[1,1] = exp(-Ctilde)*sqrt(β*c*I/N*S)
    du[2,2] = exp(-Rtilde)*sqrt(γ*I)
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [0.0,0.0]; # C,R


p = [0.05,10.0,0.25,990.0,10.0,1000.0]; # β,c,γ,S₀,I₀,N


Random.seed!(1234);


function condition(u,t,integrator,p) # Event when event_f(u,t) == 0
    (Ctilde,Rtilde) = u
    (β,c,γ,S₀,I₀,N) = p
    C = exp(Ctilde)-1.0
    R = exp(Rtilde)-1.0
    S = S₀-C
    I = I₀+C-R
    I
end


function affect!(integrator)
    terminate!(integrator)
end


cb = ContinuousCallback(
        (u,t,integrator)->condition(u,t,integrator,p),
        affect!)


prob_mbp = SDEProblem(sir_mbp!,sir_noise!,u0,tspan,p,noise_rate_prototype=A)


sol_mbp = solve(prob_mbp,
            SRA1(),
            callback=cb,
            saveat=δt);


df_mbp = DataFrame(sol_mbp(sol_mbp.t)')
df_mbp[!,:C] = exp.(df_mbp[!,:x1]) .- 1.0
df_mbp[!,:R] = exp.(df_mbp[!,:x2]) .- 1.0
df_mbp[!,:S] = p[4] .- df_mbp[!,:C]
df_mbp[!,:I] = p[5] .+ df_mbp[!,:C] .- df_mbp[!,:R]
df_mbp[!,:t] = sol_mbp.t


@df df_mbp plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark solve(prob_mbp,SRA1(),callback=cb)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

