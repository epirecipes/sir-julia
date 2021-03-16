
using DifferentialEquations
using SimpleDiffEq
using Random
using Distributions
using GpABC
using Distances
using ApproxBayes
using Plots


function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end;


tmax = 40.0
δt = 1.0
tspan = (0.0,tmax)
obstimes = 1.0:δt:tmax;
u0 = [990.0,10.0,0.0,0.0]; # S,I.R,C
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode,saveat=δt)
out_ode = Array(sol_ode)
C = out_ode[4,:]
X = C[2:end] .- C[1:(end-1)];


Random.seed!(1234)
Y = rand.(Poisson.(X));


bar(obstimes,Y)
plot!(obstimes,X)


function simdata(x)
    (i0,β) = x
    I = i0*1000.0
    prob = remake(prob_ode,u0=[1000-I,I,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    transpose(X)
end;


priors = [Uniform(0.0,0.1),Uniform(0.0,0.1)];


Yt = transpose(float.(Y));


n_particles = 2000
threshold = 80.0
sim_rej_result = SimulatedABCRejection(
    Yt, # data
    simdata, # simulator
    priors, # priors
    threshold, # threshold distance
    n_particles; # particles required
    max_iter=convert(Int, 1e7),
    distance_function = Distances.euclidean,
    write_progress=false);


plot(sim_rej_result)


n_design_points = 500
emu_rej_result = EmulatedABCRejection(Yt,
    simdata,
    priors,
    threshold,
    n_particles,
    n_design_points;
    max_iter=convert(Int, 1e7),
    distance_function = Distances.euclidean,
    write_progress=false);


plot(emu_rej_result)


threshold_schedule = [110.0,100.0,90.0,80.0];


sim_smc_result = SimulatedABCSMC(Yt,
    simdata,
    priors,
    threshold_schedule,
    n_particles;
    max_iter=convert(Int, 1e7),
    distance_function = Distances.euclidean,
    write_progress=false);


population_colors=["#FF2F4E", "#D0001F", "#A20018", "#990017"]
plot(sim_smc_result, population_colors=population_colors)


emu_smc_result = EmulatedABCSMC(Yt,
    simdata,
    priors,
    threshold_schedule,
    n_particles,
    n_design_points;
    distance_metric = Distances.euclidean,
    batch_size=1000,
    write_progress=false,
    emulator_retraining = PreviousPopulationThresholdRetraining(n_design_points, 100, 10),
    emulated_particle_selection = MeanVarEmulatedParticleSelection());


plot(emu_smc_result, population_colors=population_colors)


function simdist(x, constants, y)
  s = transpose(simdata(x))
  Distances.euclidean(s, y), 1
end;


ab_rej_setup = ABCRejection(simdist, #simulation function
  2, # number of parameters
  threshold, #target ϵ
  Prior(priors); # Prior for each of the parameters
  maxiterations = 10^7, #Maximum number of iterations before the algorithm terminates
  nparticles = n_particles
  );


ab_rej = runabc(ab_rej_setup,
            Y,
            verbose = true,
            progress = true,
            parallel = true);


plot(ab_rej)


ab_smc_setup = ABCSMC(simdist, #simulation function
  2, # number of parameters
  threshold, #target ϵ
  Prior(priors), #Prior for each of the parameters
  maxiterations=convert(Int,1e7),
  nparticles=n_particles,
  α = 0.3,
  convergence = 0.05,
  kernel = uniformkernel
  );


ab_smc = runabc(ab_smc_setup,
            Y,
            verbose = true,
            progress = true,
            parallel = true);


plot(ab_smc)

