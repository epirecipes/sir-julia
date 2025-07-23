
using Plots
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


function solve_map(f, u0, nsteps, p)
    # Pre-allocate array with correct type
    sol = similar(u0, length(u0), nsteps + 1)
    # Initialize the first column with the initial state
    sol[:, 1] = u0
    # Iterate over the time steps
    @inbounds for t in 2:nsteps+1
        u = @view sol[:, t-1] # Get the current state
        du = @view sol[:, t] # Prepare the next state
        f(du, u, p, t)       # Call the function to update du
    end
    return sol
end;


δt = 0.1 # Time step
nsteps = 400
tmax = nsteps*δt
t = 0.0:δt:tmax;


u0 = [990.0,10.0,0.0];


p = [0.05,10.0,0.25,δt]; # β,c,γ,δt


sol_map = solve_map(sir_map!, u0, nsteps, p)


S = sol_map[1,:]
I = sol_map[2,:]
R = sol_map[3,:];


plot(t,
     [S I R],
     label=["S" "I" "R"],
     xlabel="Time",
     ylabel="Number")


@benchmark solve_map(sir_map!, u0, nsteps, p)

