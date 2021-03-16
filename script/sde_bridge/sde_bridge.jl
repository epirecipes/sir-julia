
using Bridge
using StaticArrays
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


struct SIR <: ContinuousTimeProcess{SVector{3,Float64}}
    β::Float64
    c::Float64
    γ::Float64
end


function Bridge.b(t, u, P::SIR)
    (S,I,R) = u
    N = S + I + R
    dS = -P.β*P.c*S*I/N
    dI = P.β*P.c*S*I/N - P.γ*I
    dR = P.γ*I
    return @SVector [dS,dI,dR]
end


function Bridge.σ(t, u, P::SIR)
    (S,I,R) = u
    N = S + I + R
    ifrac = abs(P.β*P.c*I/N*S)
    rfrac = abs(P.γ*I)
    return @SMatrix Float64[
     sqrt(ifrac)      0.0
    -sqrt(ifrac)  -sqrt(rfrac)
     0.0   sqrt(rfrac)
    ]
end


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
ts = 0.0:δt:tmax;


u0 = @SVector [990.0,10.0,0.0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


prob = SIR(p...);


W = sample(ts, Wiener{SVector{2,Float64}}());


sol = solve(Bridge.EulerMaruyama(), u0, W, prob);


df_sde = DataFrame(Bridge.mat(sol.yy)')
df_sde[!,:t] = ts;


@df df_sde plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark begin
    W = sample(ts, Wiener{SVector{2,Float64}}());
    solve(Bridge.EulerMaruyama(), u0, W, prob);
end

