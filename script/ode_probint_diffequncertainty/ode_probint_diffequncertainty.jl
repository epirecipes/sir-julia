
using OrdinaryDiffEq
using DiffEqUncertainty
using DiffEqCallbacks
using Statistics
using Random
using Plots
using BenchmarkTools


function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*c*I/N*S
        du[2] = β*c*I/N*S - γ*I
        du[3] = γ*I
    end
    nothing
end;


function condition(u,t,integrator) # Event when event_f(u,t) == 0
  u[2]
end;


function affect!(integrator)
  integrator.u[2] = 0.0
end;


positive_cb = ContinuousCallback(condition,affect!);


tmax = 40.0
tspan = (0.0,tmax);


u0 = [990.0,10.0,0.0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


prob_ode = ODEProblem(sir_ode!, u0, tspan, p);


probint_cb_4th = AdaptiveProbIntsUncertainty(4);


num_samples = 100
ensemble_prob_ode = EnsembleProblem(prob_ode)


samples_ode = solve(ensemble_prob_ode,
                             ROS34PW3(),
                             trajectories=num_samples,
                             callback=CallbackSet(positive_cb,probint_cb_4th));


s20 = [s(20.0) for s in samples_ode]
[[mean([s[i] for s in s20]) for i in 1:3] [std([s[i] for s in s20]) for i in 1:3]]


p = plot(samples_ode[1],
     label=["S" "I" "R"],
     color=[:blue :red :green],
     xlabel="Time",
     ylabel="Number")
for i in 2:num_samples
    plot!(p,
          samples_ode[i],
          label="",
          color=[:blue :red :green])
end;


plot(p,yaxis=:log10,xlim=(15,20),ylim=(100,1000))


plot(p,yaxis=:log10,xlim=(35,40),ylim=(10,1000))


@benchmark solve(ensemble_prob_ode,
                 ROS34PW3(),
                 trajectories=100,
                 callback=CallbackSet(positive_cb,probint_cb_4th))

