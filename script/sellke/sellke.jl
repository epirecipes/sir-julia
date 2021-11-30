
using Distributions
using Random
using BenchmarkTools
using Plots


function sellke(u0, p)
    (S, I, R) = u0
    N = S+I+R
    (β, c, γ) = p
    λ = β*c/N

    Q = rand(Exponential(), S)
    sort!(Q)

    T0 = rand(Exponential(1/γ), I)
    T = rand(Exponential(1/γ), S)

    ST0 = sum(T0)
    Y = [ST0; ST0 .+ cumsum(T[1:end-1])]

    Z = findfirst(Q .> Y*λ)

    if Z === nothing
        return S+I # entire population infected
    else
        return Z+I-1
    end
end


u0 = [100,10,0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


out_sellke = map(x -> sellke(u0, p), 1:1e3)


histogram(out_sellke, bins=20, xlabel = "Final Epidemic Size", ylabel = "Frequency", legend = false)


@benchmark sellke(u0, p)

