
using DifferentialEquations
using Plots
using Random
using BenchmarkTools


function infection_rate(u,p,t)
    (S,I,R) = u
    (β,c,τ) = p
    N = S+I+R
    β*c*I/N*S
end

function infection!(integrator)
    (β,c,τ) = integrator.p
    integrator.u[1] -= 1
    integrator.u[2] += 1

    # queue recovery callback
    add_tstop!(integrator, integrator.t + τ)
end

infection_jump = ConstantRateJump(infection_rate,infection!);


function recovery_condition(u,t,integrator)
    t == integrator.tstops[1]
end

function recovery!(integrator)
	integrator.u[2] -= 1
	integrator.u[3] += 1

	reset_aggregated_jumps!(integrator)
    popfirst!(integrator.tstops)
    integrator.tstops_idx -= 1
end

recovery_callback = DiscreteCallback(recovery_condition, recovery!, save_positions = (false, false))


tmax = 40.0
tspan = (0.0,tmax);


δt = 0.1
t = 0:δt:tmax;


u0 = [990,10,0]; # S,I,R


p = [0.05,10.0,4.0]; # β,c,τ


Random.seed!(1234);


prob = DiscreteProblem(u0,tspan,p);


prob_jump = JumpProblem(prob, Direct(), infection_jump);


sol_jump = solve(prob_jump,SSAStepper(), callback = recovery_callback);


out_jump = sol_jump(t);


plot(
    out_jump,
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number"
)


@benchmark solve(prob_jump,SSAStepper(), callback = recovery_callback);

