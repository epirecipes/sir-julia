
using Random
using Distributions
using Pathogen
using Plots
using Plots.PlotMeasures
using StatsPlots;


Random.seed!(1);


N = 100
I₀ = 1
locations = DataFrame(x = rand(Uniform(0, 10), N),
                  y = rand(Uniform(0, 10), N))

# Precalculate distances
dists = [1.0 for i = 1:N, j = 1:N]
# Set diagonal to zero
[dists[i,i] = 0.0 for i in 1:N]

pop = Population(locations, dists)

function _constant(params::Vector{Float64}, pop::Population, i::Int64)
    return params[1]
end

function _one(params::Vector{Float64}, pop::Population, i::Int64)
    return 1.0
end

function _zero(params::Vector{Float64}, pop::Population, i::Int64)
    return 0.0
end

function _one(params::Vector{Float64}, pop::Population, i::Int64, k:: Int64)
    return 1.0
end

rf = RiskFunctions{SIR}(_zero, # sparks function
                        _one, # susceptibility function
                        _one, # infectivity function: defines a distance
                        _constant, # transmissability function
                        _constant) # removal function

# Parametrize risk functions for simulation
rparams = RiskParameters{SIR}(Float64[], # sparks function parameter(s)
                              Float64[], # susceptibility function parameter(s)
                              Float64[], # infectivity function parameter(s)
                              [0.5/N], # transmissibility function parameter(s) βc/N
                              [0.25]) # removal function parameter(s) γ

# Set starting states in population
# Set first I₀ individuals as infectious, others as susceptible to start
starting_states = [fill(State_I, I₀); fill(State_S, N-I₀)]
# Initialize simulation
sim = Simulation(pop, starting_states, rf, rparams)
# Simulate
simulate!(sim, tmax=40.0);


p1 = plot(sim.events, 0.0, 40.0, legendfont=font(6), xaxis=font(10), bottom_margin=30px)
# Population/TransmissionNetwork plots
p2=plot(sim.transmission_network, sim.population, sim.events, 0.0, title="Time = 0", titlefontsize = 8)
p3=plot(sim.transmission_network, sim.population, sim.events, 10.0, title="Time = 10", titlefontsize = 8)
p4=plot(sim.transmission_network, sim.population, sim.events, 20.0, title="Time = 20", titlefontsize = 8)
p5=plot(sim.transmission_network, sim.population, sim.events, 30.0, title="Time = 30", titlefontsize = 8)
p6=plot(sim.transmission_network, sim.population, sim.events, 40.0, title="Time = 40", titlefontsize = 8)
# Combine
l = @layout [a
             grid(1,5)]
combinedplots1 = plot(p1, p2, p3, p4, p5, p6, layout=l)


obs = observe(sim, Dirac(0.0), Dirac(0.0), force=true);


rpriors = RiskPriors{SIR}(UnivariateDistribution[],
                          UnivariateDistribution[],
                          UnivariateDistribution[],
                          [Uniform(0.001, 0.01)],
                          [Uniform(0.1, 0.5)]);


ee = EventExtents{SIR}(0.001, 0.001);


mcmc = MCMC(obs, ee, pop, starting_states, rf, rpriors)
start!(mcmc, attempts=10000); # 1 chain, with 10k initialization attempts


niter = 11000
burnin = 1000
thin = 20
iterate!(mcmc, niter, 1.0, condition_on_network=true, event_batches=5);


summary(mcmc, burnin=burnin, thin=thin)


β = [mcmc.markov_chains[1].risk_parameters[i].transmissibility[1] for i in (burnin+1):niter]
γ = [mcmc.markov_chains[1].risk_parameters[i].removal[1] for i in (burnin+1):niter];


marginalkde(β, γ)


# MCMC trace plot
p1 = plot(1:thin:niter,
  mcmc.markov_chains[1].risk_parameters, yscale=:log10, title="TN-ILM parameters", xguidefontsize=8, yguidefontsize=8, xtickfontsize=7, ytickfontsize=7, titlefontsize=11, bottom_margin=30px)

# State plots
p2 = plot(mcmc.markov_chains[1].events[niter], State_S,
          linealpha=0.01, title="S", xguidefontsize=8, yguidefontsize=8,
          xtickfontsize=7, ytickfontsize=7, titlefontsize=11)
for i=(burnin+1):thin:niter
  plot!(p2, mcmc.markov_chains[1].events[i], State_S, linealpha=0.01)
end
plot!(p2, sim.events, State_S, linecolor=:black, linewidth=1.5)

p3 = plot(mcmc.markov_chains[1].events[niter], State_I,
          linealpha=0.01, title="I", xguidefontsize=8, yguidefontsize=8, xtickfontsize=7, ytickfontsize=7, titlefontsize=11)
for i=(burnin+1):thin:niter
  plot!(p3, mcmc.markov_chains[1].events[i], State_I, linealpha=0.01)
end
plot!(p3, sim.events, State_I, linecolor=:black, linewidth=1.5)

p4 = plot(mcmc.markov_chains[1].events[niter], State_R,
          linealpha=0.01, title="R", xguidefontsize=8, yguidefontsize=8, xtickfontsize=7, ytickfontsize=7, titlefontsize=11)
for i=(burnin+1):thin:niter
  plot!(p4, mcmc.markov_chains[1].events[i], State_R, linealpha=0.01)
end
plot!(p4, sim.events, State_R, linecolor=:black, linewidth=1.5)
# Combine
l = @layout [a; [b c d]]
combinedplots2 = plot(p1, p2, p3, p4, layout=l)


p1 = plot(sim.transmission_network, sim.population, title="True Transmission\nNetwork", titlefontsize=11, framestyle=:box)

tnp = TransmissionNetworkPosterior(mcmc, burnin=burnin, thin=thin)
p2 = plot(tnp, sim.population, title="Transmission Network\nPosterior Distribution", titlefontsize=11, framestyle=:box)

combinedplots3 = plot(p1, p2, layout=(1, 2))

