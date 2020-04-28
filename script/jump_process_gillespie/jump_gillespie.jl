
using Pkg
pkg"add https://github.com/sdwfrost/Gillespie.jl"


using Gillespie
using Random
using StatsPlots
using BenchmarkTools


function sir_rates(x,parms)
  (S,I,R) = x
  (β,c,γ) = parms
  N = S+I+R
  infection = β*c*I/N*S
  recovery = γ*I
  [infection,recovery]
end


sir_transitions = [[-1 1 0];[0 -1 1]]


tmax = 40.0


u0 = [990,10,0] # S,I,R


p = [0.05,10.0,0.25] # β,c,γ


Random.seed!(1234);


sir_result = ssa(u0,sir_rates,sir_transitions,p,tmax)


sir_data = ssa_data(sir_result)


@df sir_data plot(:time,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark ssa(u0,sir_rates,sir_transitions,p,tmax)


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

