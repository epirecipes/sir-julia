
using DifferentialEquations
using OrdinaryDiffEq
using Distributions
using MonteCarloMeasurements
using StatsBase
using Plots


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


δt = 1.0
tmax = 40.0
tspan = (0.0,tmax);


N = 1000.0;


n_samples = 1000; # Number of samples


p = [Particles(n_samples,Uniform(0.01,0.1)),
      Particles(n_samples,Uniform(5,20.0)),
      Particles(n_samples,Uniform(0.1,1.0))]


I₀=Particles(n_samples,Uniform(1.0,50.0))
u0 = [N-I₀,I₀,0.0]


prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


sol_ode = solve(prob_ode, Tsit5(), dt=δt);


s20 = sol_ode(20.0)


l = @layout [a b c]
binwidth = 50
pl1 = histogram(s20[1],bins=0:binwidth:N, title="S(20)", xlabel="S", ylabel="Frequency", color=:blue)
pl2 = histogram(s20[2],bins=0:binwidth:N, title="I(20)", xlabel="I", ylabel="Frequency", color=:red)
pl3 = histogram(s20[3],bins=0:binwidth:N, title="R(20)", xlabel="R", ylabel="Frequency", color=:green)
plot(pl1,pl2,pl3,layout=l,legend=false)


corkendall(hcat(Array(p),Array(I₀)),Array(s20))

