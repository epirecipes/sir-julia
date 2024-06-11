
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools


C_code = """
void sir_ode(double *du, double *u, double *p, double *t){
    double β = p[0];
    double c = p[1];
    double γ = p[2];
    double S = u[0];
    double I = u[1];
    double R = u[2];
    double N = S + I + R;
    du[0] = -β*c*S*I/N;
    du[1] = β*c*S*I/N - γ*I;
    du[2] = γ*I;
}
""";


const Clib = tempname();


open(`gcc -fPIC -O3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`, "w") do f
    print(f, C_code)
end;


open(Clib * "." * "c", "w") do f
    write(f, C_code)
end
run(`gcc -fPIC -O3 -shared -o $(Clib * "." * Libdl.dlext) $(Clib * "." * "c")`);


function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Clib,), Cvoid,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), du, u, p, Ref(t))
end;


sir_ode = ODEFunction(sir_ode_jl!);


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);


u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem{true}(sir_ode, u0, tspan, p)
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

