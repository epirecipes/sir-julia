
using JuMP
using Ipopt
using Plots;


β = 0.5 # infectivity rate
γ = 0.25 # recovery rate
υ_max = 0.5 # maximum intervention
I_max = 0.1 # maximum allowable infectives at one time
silent = true;


t0 = 0.0 # start time
tf = 100.0 # final time
δt = 0.1 # timestep
T = Int(tf/δt); # number of timesteps


S₀ = 0.99
I₀ = 0.01
C₀ = 0.00;


model = Model(Ipopt.Optimizer)


@variable(model, S[1:(T+1)])
@variable(model, I[1:(T+1)])
@variable(model, C[1:(T+1)])
@variable(model, υ[1:(T+1)])
@variable(model, υ_total);


# Initial conditions
@constraint(model, S[1]==S₀)
@constraint(model, I[1]==I₀)
@constraint(model, C[1]==C₀)

# Constraints on variables
@constraint(model, [t=2:(T+1)], 0 ≤  S[t] ≤ 1)
@constraint(model, [t=2:(T+1)], 0 ≤  I[t] ≤ I_max)
@constraint(model, [t=2:(T+1)], 0 ≤  C[t] ≤ 1);


@constraint(model, [t=1:(T+1)], 0 ≤  υ[t] ≤ υ_max);
@constraint(model, δt*sum(υ) == υ_total);


@NLexpression(model, infection[t=1:T], (1-exp(-(1 - υ[t]) * β * I[t] * δt)) * S[t])
@NLexpression(model, recovery[t=1:T], (1-exp(-γ*δt)) * I[t]);


@NLconstraint(model, [t=1:T], S[t+1] == S[t] - infection[t])
@NLconstraint(model, [t=1:T], I[t+1] == I[t] + infection[t] - recovery[t])
@NLconstraint(model, [t=1:T], C[t+1] == C[t] + infection[t]);


@objective(model, Min, υ_total);


if silent
    set_silent(model)
end
optimize!(model)


termination_status(model)


S_opt = value.(S)
I_opt = value.(I)
C_opt = value.(C)
υ_opt = value.(υ)
Rₜ_opt = β * S_opt/γ # absence of intervention
Rₜ′_opt = Rₜ_opt .* (1 .- υ_opt) # in presence of intervention
ts = collect(0:δt:tf);


using DataInterpolations
using NonlinearSolve
Rₜ_interp = CubicSpline(Rₜ_opt,ts)
f(u, p) = [Rₜ_interp(u[1]) - 1.0]
u0 = [(tf-t0)/2]
Rtprob = NonlinearProblem(f, u0)
Rtsol = solve(Rtprob, NewtonRaphson(), abstol = 1e-9).u[1];


plot(ts, S_opt, label="S", xlabel="Time", ylabel="Number", legend=:right, xlim=(0,60))
plot!(ts, I_opt, label="I")
plot!(ts, C_opt, label="C")
plot!(ts, υ_opt, label="Optimized υ")
hline!([I_max], color=:gray, alpha=0.5, label="Threshold I")
hline!([υ_max], color=:orange, alpha=0.5, label="Threshold υ")


plot(ts, Rₜ_opt, label="Rₜ", xlabel="Time", ylabel="Number", legend=:right, xlim=(0,60))
plot!(ts, Rₜ′_opt, label="Rₜ including policy")
plot!(ts, υ_opt, label="Optimized υ")
vline!([Rtsol], color=:gray, alpha=0.5, label=false)
hline!([1.0], color=:gray, alpha=0.5, label=false)

