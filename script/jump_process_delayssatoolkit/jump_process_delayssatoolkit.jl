
using ModelingToolkit
using Catalyst
using JumpProcesses
using DelaySSAToolkit
using Random
using Distributions
using Plots
using BenchmarkTools;


Random.seed!(1234);


x = 0:0.1:10
ye = pdf.(Exponential(4),x)
yg = pdf.(Gamma(4,1),x)
plot(x, ye, xlabel="Recovery time", ylabel="Density", label="Exponential(4)")
plot!(x, yg, label="Gamma(4,1)")


@parameters t β
@variables S(t) I(t) R(t)
N = S + I + R
rxs = [Reaction(β/N, [S,I], [I], [1,1], [2])];


@named rs  = ReactionSystem(rxs, t, [S,I,R], [β])
jsys = convert(JumpSystem, rs, combinatoric_ratelaws=false);


species(jsys), length(reactions(jsys))


recovery_trigger_affect! = function (integrator, rng, dist)
    # Here, τ is the delay until one of the delay channels
    # is triggered  
    τ = rand(rng, dist)
    # There is only one delay channel in this example
    append!(integrator.de_chan[1], τ)
end;


tmax = 40.0
tspan = (0.0, tmax);
p = [0.5]  # β   
dist = Gamma(4.0, 1.0)
u0 = [990, 10, 0] # S,I,R
u0_delay = [rand(dist) for i in 1:u0[2]];


# Reaction 1 (infection) triggers the above callback
delay_trigger = Dict(1=> (i, r) -> recovery_trigger_affect!(i, r, dist))
# There are no interrupts in the system
delay_interrupt = Dict()
# After the delay, increment state 3 (R) by 1 and decrement state 2 (I) by 1
delay_complete = Dict(1=>[3=>1, 2=>-1])
# Combine the above `Dict`s in a `DelayJumpSet`
delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt);


dprob = DiscreteProblem(jsys, u0, tspan, p);


de_chan0 = [u0_delay];


djprob = DelayJumpProblem(jsys, dprob, DelayRejection(), delayjumpset, de_chan0, save_positions=(true,true));


djsol = solve(djprob, SSAStepper());


plot(djsol, xlabel="Time", ylabel="Number")


@benchmark solve(djprob, SSAStepper())

