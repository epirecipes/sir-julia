
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools


FPC_code = """
library sir_ode_lib;

type
  PDouble = ^Double;
  TDoubleArray = array[0..2] of Double;
  PDoubleArray = ^TDoubleArray;

procedure sir_ode(du, u, p: PDoubleArray; t: PDouble); cdecl; export;
var
  beta, c, gamma, S, I, R, N: Double;
begin
  // Access the elements of the arrays through pointers
  beta := p^[0];
  c := p^[1];
  gamma := p^[2];
  S := u^[0];
  I := u^[1];
  R := u^[2];
  N := S + I + R;
  
  du^[0] := -beta * c * S * I / N;
  du^[1] := beta * c * S * I / N - gamma * I;
  du^[2] := gamma * I;
end;

exports
  sir_ode;

begin
end.
""";


const FPClib = tempname()
open(FPClib * "." * "pas", "w") do f
    write(f, FPC_code)
end
run(`fpc -Cg -XS -o$(FPClib * "." * Libdl.dlext) $(FPClib * "." * "pas")`);


function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,FPClib,), Cvoid,
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

