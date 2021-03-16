
using DifferentialEquations
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


δt = 0.1
tmax = 80.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0]; # S,I.R


p = [0.05,10.0,0.25]; # β,c,γ


lockdown_times = [10.0, 20.0]
condition(u,t,integrator) = t ∈ lockdown_times
function affect!(integrator)
    if integrator.t < lockdown_times[2]
        integrator.p[1] = 0.01
    else
        integrator.p[1] = 0.05
    end
end
cb = PresetTimeCallback(lockdown_times, affect!);


prob_ode = ODEProblem(sir_ode!,u0,tspan,p)


sol_ode = solve(prob_ode, callback = cb);


plot(sol_ode, label = ["S" "I" "R"], title = "Lockdown in a SIR model")
vline!(lockdown_times, c = :red, w = 2, label = "")

