
using OrdinaryDiffEq
using SimpleDiffEq
using Plots
using BenchmarkTools;


@inline function rate_to_proportion(r,t)
    1-exp(-r*t)
end;


function sir_map!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ,δt) = p
    N = S+I+R
    infection = rate_to_proportion(β*c*I/N,δt)*S
    recovery = rate_to_proportion(γ,δt)*I
    @inbounds begin
        du[1] = S-infection
        du[2] = I+infection-recovery
        du[3] = R+recovery
    end
    nothing
end;


δt = 0.1
nsteps = 400
tmax = nsteps*δt
tspan = (0.0,nsteps)
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0];


p = [0.05,10.0,0.25,δt]; # β,c,γ,δt


prob_map = DiscreteProblem(sir_map!,u0,tspan,p);


sol_map = solve(prob_map,FunctionMap());


# Extract state variables
t = sol_map.t
S = [u[1] for u in sol_map.u]
I = [u[2] for u in sol_map.u]
R = [u[3] for u in sol_map.u];


plot(t,
     [S I R],
     label=["S" "I" "R"],
     xlabel="Time",
     ylabel="Number")


@benchmark solve(prob_map, FunctionMap())


@benchmark solve(prob_map, SimpleFunctionMap())

