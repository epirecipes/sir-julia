
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools


Rust_code = """
#![allow(non_snake_case)]
#[no_mangle]
pub extern "C" fn sir_ode(du: &mut [f64; 3], u: &[f64; 3], p: &[f64; 3], _t: f64) {
    let beta = p[0];
    let c = p[1];
    let gamma = p[2];
    let S = u[0];
    let I = u[1];
    let R = u[2];
    let N = S + I + R;
    du[0] = -beta * c * S * I / N;
    du[1] = beta * c * S * I / N - gamma * I;
    du[2] = gamma * I;
}
""";


const Rustlib = tempname();


open(Rustlib * "." * "rs", "w") do f
    write(f, Rust_code)
end
run(`rustc --crate-type cdylib -o $(Rustlib * "." * Libdl.dlext) $(Rustlib * "." * "rs")`);


function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Rustlib,), Cvoid,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), du, u, p, Ref(t))
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);


u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem{true}(sir_ode_jl!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), dt = δt);


plot(sol_ode, labels=["S" "I" "R"], lw = 2, xlabel = "Time", ylabel = "Number")


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

