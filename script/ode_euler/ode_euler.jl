
using Tables
using DataFrames
using StatsPlots
using BenchmarkTools


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


t0 = 0.0
δt = 0.1
tmax = 40.0
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


function euler(f, u0, p, δt, t0, tmax)
    t = t0 # Initialize time
    u = copy(u0) # Initialize struct parametric inherited
    du = zeros(length(u0)) # Initialize derivatives
    f(du,u,p,t)
    sol = [] # Store output
    times = [] # Store times
    push!(sol,copy(u))
    push!(times,t)
    # Main loop
    while t < tmax
        t = t + δt # Update time
        u .= u .+ du.*δt # Update state
        sir_ode!(du,u,p,t) # Update derivative
        push!(sol,copy(u)) # Store output
        push!(times,t) # Store time
    end
    sol = hcat(sol...) # Convert to matrix
    return times, sol
end;


times, sol = euler(sir_ode!, u0, p, δt, t0, tmax);


df = DataFrame(Tables.table(sol'))
rename!(df,["S","I","R"])
df[!,:t] = times;


@df df plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark euler(sir_ode!, u0, p, δt, t0, tmax)

