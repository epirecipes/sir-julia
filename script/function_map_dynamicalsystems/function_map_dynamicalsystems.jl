
using DynamicalSystems
using DataFrames
using StatsPlots
using BenchmarkTools


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
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0];


p = [0.05,10.0,0.25,δt]; # β,c,γ,δt


ds = DiscreteDynamicalSystem(sir_map!, u0, p, t0 = 0)


sol = trajectory(ds,nsteps)


df = DataFrame(Matrix(sol))
df[!,:t] = t;


@df df plot(:t,
    [:x1, :x2, :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark trajectory(ds,nsteps)

