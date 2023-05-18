
using JuMP
using Ipopt
using Plots;


β = 0.5 # infectivity rate
γ = 0.25 # recovery rate
υ_max = 0.05 # maximum vaccination rate 
υ_total = 1.0 # maximum cost
silent = true; # whether to display output of solver


t0 = 0.0
tf = 100.0
dt = 1.0;


S₀ = 0.99
I₀ = 0.01
C₀ = 0.00;


model = Model(Ipopt.Optimizer)


T = Int(tf/dt)
@variable(model, S[1:(T+1)])
@variable(model, I[1:(T+1)])
@variable(model, C[1:(T+1)])
@variable(model, υ[1:(T+1)]);


# Initial conditions
@constraint(model, S[1]==S₀)
@constraint(model, I[1]==I₀)
@constraint(model, C[1]==C₀)

# Constraints on variables
@constraint(model, [t=2:(T+1)], 0 ≤  S[t] ≤ 1)
@constraint(model, [t=2:(T+1)], 0 ≤  I[t] ≤ 1)
@constraint(model, [t=2:(T+1)], 0 ≤  C[t] ≤ 1);


@constraint(model, [t=1:(T+1)], 0 ≤  υ[t] ≤ υ_max)
@constraint(model, [t=1:(T+1)], sum(dt*υ[t]*S[t]) ≤ υ_total);


@NLexpression(model, infection[t=1:T], (1 - exp(-β*I[t]*dt)) * S[t])
@NLexpression(model, recovery[t=1:T], (1-exp(-γ*dt)) * I[t]);
@NLexpression(model, vaccination[t=1:T], (1-exp(-υ[t]*dt)) * S[t]);


@NLconstraint(model, [t=1:T], S[t+1] == S[t] - infection[t] - vaccination[t])
@NLconstraint(model, [t=1:T], I[t+1] == I[t] + infection[t] - recovery[t])
@NLconstraint(model, [t=1:T], C[t+1] == C[t] + infection[t]);


@objective(model, Min, C[T+1]);


if silent
    set_silent(model)
end
optimize!(model)


termination_status(model)


S_opt = value.(S)
I_opt = value.(I)
C_opt = value.(C)
υ_opt = value.(υ)
ts = collect(0:dt:tf);


plot(ts, S_opt, label="S", xlabel="Time", ylabel="Number")
plot!(ts, I_opt, label="I")
plot!(ts, C_opt, label="C")
plot!(ts, υ_opt, label="Optimized υ")


model = Model(Ipopt.Optimizer)
@variable(model, S[1:(T+1)])
@variable(model, I[1:(T+1)])
@variable(model, C[1:(T+1)])
@variable(model, υ[1:(T+1)])

# Initial conditions
@constraint(model, S[1]==S₀)
@constraint(model, I[1]==I₀)
@constraint(model, C[1]==C₀)

# Constraints on variables
@constraint(model, [t=2:(T+1)], 0 ≤  S[t] ≤ 1)
@constraint(model, [t=2:(T+1)], 0 ≤  I[t] ≤ 1)
@constraint(model, [t=2:(T+1)], 0 ≤  C[t] ≤ 1)

@constraint(model, [t=1:(T+1)], 0 ≤  υ[t] ≤ υ_max)
# The below constraint is changed from the per-capita example
@constraint(model, sum(dt*υ) ≤ υ_total)

@NLexpression(model, infection[t=1:T], (1 - exp(-β*I[t]*dt)) * S[t])
@NLexpression(model, recovery[t=1:T], (1-exp(-γ*dt)) * I[t])
# The below constraint is changed from the per-capita example
@NLexpression(model, vaccination[t=1:T], υ[t]*dt)

@NLconstraint(model, [t=1:T], S[t+1] == S[t] - infection[t] - vaccination[t])
@NLconstraint(model, [t=1:T], I[t+1] == I[t] + infection[t] - recovery[t])
@NLconstraint(model, [t=1:T], C[t+1] == C[t] + infection[t])

@objective(model, Min, C[T+1]);


if silent
    set_silent(model)
end
optimize!(model)


S_opt = value.(S)
I_opt = value.(I)
C_opt = value.(C)
υ_opt = value.(υ)
ts = collect(0:dt:tf);


plot(ts, S_opt, label="S", xlabel="Time", ylabel="Number")
plot!(ts, I_opt, label="I")
plot!(ts, C_opt, label="C")
plot!(ts, υ_opt, label="Optimized υ")

