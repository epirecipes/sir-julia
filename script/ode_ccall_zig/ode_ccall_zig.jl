
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools


Zig_code = """
export fn sir_ode(du: [*c]f64, u: [*c]const f64, p: [*c]const f64, t: [*c]const f64) void {
    const beta: f64 = p[0];
    const c: f64 = p[1];
    const gamma: f64 = p[2];
    const S: f64 = u[0];
    const I: f64 = u[1];
    const R: f64 = u[2];
    const N: f64 = S + I + R;
    _ = t;

    du[0] = -beta * c * S * I / N;
    du[1] = beta * c * S * I / N - gamma * I;
    du[2] = gamma * I;
}
""";


const Ziglib = tempname();
open(Ziglib * "." * "zig", "w") do f
    write(f, Zig_code)
end
run(`zig build-lib -dynamic -O ReleaseSafe -fPIC -femit-bin=$(Ziglib * "." * Libdl.dlext) $(Ziglib * "." * "zig")`);


function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Ziglib,), Cvoid,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), du, u, p, Ref(t))
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem{true}(sir_ode_jl!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), dt = δt);


plot(sol_ode)


@benchmark solve(prob_ode, Tsit5(), dt = δt)


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
end
prob_ode2 = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode2 = solve(prob_ode2, Tsit5(), dt = δt)
@benchmark solve(prob_ode2, Tsit5(), dt = δt)

