
using Random
using Distributions
using Pathogen
using Plots
using Plots.PlotMeasures
using BenchmarkTools;


Random.seed!(1234);


N = 1000;


locations = DataFrame(x = rand(Uniform(0, 10), N),
                      y = rand(Uniform(0, 10), N));


dists = [1.0 for i = 1:N, j = 1:N]
# Set diagonal to zero
[dists[i,i] = 0.0 for i in 1:N]
pop = Population(locations, dists);


function _constant(params::Vector{Float64}, pop::Population, i::Int64)
    return params[1]
end

function _one(params::Vector{Float64}, pop::Population, i::Int64)
    return 1.0
end

function _one(params::Vector{Float64}, pop::Population, i::Int64, k:: Int64)
    return 1.0
end

function _zero(params::Vector{Float64}, pop::Population, i::Int64)
    return 0.0
end;


rf = RiskFunctions{SIR}(_zero, # sparks function
                        _one, # susceptibility function
                        _one, # infectivity function: defines a distance
                        _constant, # transmissability function
                        _constant); # removal function


rparams = RiskParameters{SIR}(Float64[], # sparks function parameter(s)
                              Float64[], # susceptibility function parameter(s)
                              Float64[], # infectivity function parameter(s)
                              [0.5/N], # transmissibility function parameter(s)
                              [0.25]); # removal function parameter(s)


I₀ = 10
starting_states = [fill(State_I, I₀); fill(State_S, N-I₀)];


sim = Simulation(pop, starting_states, rf, rparams);


simulate!(sim, tmax=40.0);


plot(sim.events, 0.0, 40.0)


@benchmark begin
sim = Simulation(pop, starting_states, rf, rparams)
simulate!(sim, tmax=40.0)
end

