
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools


Nim_code = """
proc sir_ode*(du: ptr array[3, float64], u: ptr array[3, float64], p: ptr array[3, float64], t: ptr float64) {.exportc, dynlib, cdecl.} =
  let
    beta = p[0]
    c = p[1]
    gamma = p[2]
    S = u[0]
    I = u[1]
    R = u[2]
    N = S + I + R

  du[0] = -beta * c * S * I / N
  du[1] = beta * c * S * I / N - gamma * I
  du[2] = gamma * I
""";


const Nimlib = tempname();
open(Nimlib * "." * "nim", "w") do f
    write(f, Nim_code)
end
run(`nim c -d:release --app:lib --noMain --gc:none -d:release -o:$(Nimlib * "." * Libdl.dlext) $(Nimlib * "." * "nim")`);


function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Nimlib,), Cvoid,
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

