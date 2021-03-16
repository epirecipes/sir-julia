
using DifferentialEquations
using SimpleDiffEq
using DiffEqCallbacks
using Random
using Distributions
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
sol_ode_cumulative = solve(prob_ode,Tsit5(),saveat=δt);


out = Array(sol_ode_cumulative)
C = out[4,:];


X = C[2:end] .- C[1:(end-1)];


Random.seed!(1234);


Y = rand.(Poisson.(X));


bar(obstimes,Y)
plot!(obstimes,X)


S = out[1,:]
Cpred = 990.0 .- S
Cdiff = Cpred .- C
plot(obstimes,Cdiff[2:end])


affect!(integrator) = integrator.u[4] = 0.0
cb_zero = PresetTimeCallback(obstimes,affect!);


sol_ode_cb = solve(prob_ode,Tsit5(),saveat=δt,callback=cb_zero);


X_cb = sol_ode_cb(obstimes)[4,:];


Random.seed!(1234);


Y_cb = rand.(Poisson.(X_cb));


X_diff_cb = X_cb .- X
plot(obstimes,X_diff_cb)


Y_diff_cb = Y_cb .- Y
plot(obstimes,Y_diff_cb)


function sir_dde!(du,u,h,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    e = oneunit(t)
    history = h(p, t-e)*inv(e)
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection - history[4]
    end
    nothing
end;


function sir_history(p, t; idxs = 5)
    zero(t)
end;


prob_dde = DDEProblem(DDEFunction(sir_dde!),
        u0,
        sir_history,
        tspan,
        p;
        constant_lags = [1.0]);


sol_dde = solve(prob_dde,MethodOfSteps(Tsit5()));


X_dde = sol_dde(obstimes)[4,:];


Random.seed!(1234)
Y_dde = rand.(Poisson.(X_dde));


X_diff_dde = X_dde .- X
plot(X_diff_dde)


Y_diff_dde = Y_dde .- Y
plot(obstimes, Y_diff_dde)

