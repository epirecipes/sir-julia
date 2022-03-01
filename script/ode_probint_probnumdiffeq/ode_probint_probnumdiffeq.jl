
using ProbNumDiffEq
using Random
using Statistics
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


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);


u0 = [990.0,10.0,0.0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


sol_ode_ek0 = solve(prob_ode,
                EK0(prior=:ibm, order=3, diffusionmodel=DynamicDiffusion(), smooth=true),
                dt=δt,
                abstol=1e-1,
                reltol=1e-2);


sol_ode_ek1 = solve(prob_ode,
                EK1(prior=:ibm, order=3, diffusionmodel=DynamicDiffusion(), smooth=true),
                dt=δt,
                abstol=1e-1,
                reltol=1e-2);


s20_ek0 = sol_ode_ek0(20.0)
[mean(s20_ek0) std(s20_ek0)]


s20_ek1 = sol_ode_ek1(20.0)
[mean(s20_ek1) std(s20_ek1)]


num_samples = 100
samples_ode_ek0 = ProbNumDiffEq.sample(sol_ode_ek0, num_samples);
samples_ode_ek1 = ProbNumDiffEq.sample(sol_ode_ek1, num_samples);


p_ek0 = plot(sol_ode_ek0.t,
         samples_ode_ek0[:, :, 1],
         label=["S" "I" "R"],
         color=[:blue :red :green],
         xlabel="Time",
         ylabel="Number",
         title="EK0")
for i in 2:num_samples
    plot!(p_ek0,
          sol_ode_ek0.t,
          samples_ode_ek0[:, :, i],
          label="",
          color=[:blue :red :green])
end;


p_ek1 = plot(sol_ode_ek1.t,
         samples_ode_ek1[:, :, 1],
         label=["S" "I" "R"],
         color=[:blue :green],
         xlabel="Time",
         ylabel="Number",
         title="EK1")
for i in 2:num_samples
    plot!(p_ek1,
          sol_ode_ek1.t,
          samples_ode_ek1[:, :, i],
          label="",
          color=[:blue :red :green],)
end;


plot(p_ek0, p_ek1, layout = (1,2), xlim=(15,20),ylim=(100,1000),yaxis=:log10)


plot(p_ek0, p_ek1, layout = (1,2), xlim=(35,40),ylim=(10,1000),yaxis=:log10)


@benchmark solve(prob_ode,
                 EK0(prior=:ibm, order=3, diffusionmodel=DynamicDiffusion(), smooth=true),
                 abstol=1e-1,
                 reltol=1e-2)


@benchmark solve(prob_ode,
                 EK1(prior=:ibm, order=3, diffusionmodel=DynamicDiffusion(), smooth=true),
                 abstol=1e-1,
                 reltol=1e-2)

