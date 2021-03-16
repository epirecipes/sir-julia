
using DifferentialEquations
using SimpleDiffEq
using DiffEqSensitivity
using Random
using Distributions
using DiffEqParamEstim
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


δt = 1.0
tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax;


u0 = [990.0,10.0,0.0,0.0]; # S,I.R,Y


p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode,Tsit5(),saveat=δt);


out = Array(sol_ode)
C = out[4,:];


X = C[2:end] .- C[1:(end-1)];


Random.seed!(1234);


Y = rand.(Poisson.(X));


using Optim


function ss1(β)
    prob = remake(prob_ode,u0=[990.0,10.0,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    return(sum((X .- Y) .^2))
end;


function nll1(β)
    prob = remake(prob_ode,u0=[990.0,10.0,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -sum(logpdf.(Poisson.(X),Y))
end;


lower1 = 0.0
upper1 = 1.0
initial_x1 = 0.1;


opt1_ss = Optim.optimize(ss1,lower1,upper1)


opt1_nll = Optim.optimize(nll1,lower1,upper1)


function ss2(x)
    (i0,β) = x
    I = i0*1000.0
    prob = remake(prob_ode,u0=[1000.0-I,I,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    return(sum((X .- Y) .^2))
end;


function nll2(x)
    (i0,β) = x
    I = i0*1000.0
    prob = remake(prob_ode,u0=[1000.0-I,I,0.0,0.0],p=[β,10.0,0.25])
    sol = solve(prob,Tsit5(),saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -sum(logpdf.(Poisson.(X),Y))
end;


lower2 = [0.0,0.0]
upper2 = [1.0,1.0]
initial_x2 = [0.01,0.1];


opt2_ss = Optim.optimize(ss2,lower2,upper2,initial_x2)


opt2_nll = Optim.optimize(nll2,lower2,upper2,initial_x2)


function loss_function(sol)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -sum(logpdf.(Poisson.(X),Y))
end;


prob_generator = (prob,q) -> remake(prob,
    u0=[1000.0-(q[1]*1000),q[1]*1000,0.0,0.0],
    p=[q[2],10.0,0.25]);


cost_function = build_loss_objective(prob_ode,
    Tsit5(),
    loss_function,
    saveat=δt,
    prob_generator = prob_generator,
    maxiters=100,
    verbose=false);


opt_pe1 = Optim.optimize(cost_function,lower2,upper2,initial_x2)


using NLopt
opt = Opt(:LD_MMA, 2)
opt.lower_bounds = lower2
opt.upper_bounds = upper2
opt.min_objective = cost_function
opt.maxeval = 10000
(minf,minx,ret) = NLopt.optimize(opt,initial_x2)


using BlackBoxOptim
bound1 = Tuple{Float64, Float64}[(0.0,1.0),(0.0, 1.0)]
result = bboptimize(cost_function;SearchRange = bound1, MaxSteps = 1e4)

