
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools


F90_code = """
module sir_module
    use iso_c_binding

    contains

    subroutine sir_ode(du, u, p, t) bind(c)
        real(c_double), intent(out) :: du(3)
        real(c_double), intent(in) :: u(3)
        real(c_double), intent(in) :: p(3)
        real(c_double), intent(in) :: t

        real(c_double) :: beta, c, gamma, S, I, R, N

        beta = p(1)
        c = p(2)
        gamma = p(3)
        S = u(1)
        I = u(2)
        R = u(3)
        N = S + I + R

        du(1) = -beta*c*S*I/N
        du(2) = beta*c*S*I/N - gamma*I
        du(3) = gamma*I
    end subroutine sir_ode

end module sir_module
""";


const Flib = tempname();


open(Flib * "." * "f90", "w") do f
    write(f, F90_code)
end
run(`gfortran -fPIC -shared -O3 -o $(Flib * "." * Libdl.dlext) $(Flib * "." * "f90")`);


function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Flib,), Cvoid,
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

