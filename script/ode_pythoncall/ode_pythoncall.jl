
using OrdinaryDiffEq
using PythonCall
using Plots
using BenchmarkTools


@pyexec """
def sir_ode_op_py(u,p,t):
    S = u[0]
    I = u[1]
    R = u[2]
    N = S+I+R
    beta = p[0]
    c = p[1]
    gamma = p[2]
    dS = -beta*c*I/N*S
    dI = beta*c*I/N*S - gamma*I
    dR = gamma*I
    return [dS,dI,dR]
""" => sir_ode_op_py;


sir_ode_op_jl(u,p,t) = pyconvert(Array, sir_ode_op_py(u, p, t));


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode_op = ODEProblem{false}(sir_ode_op_jl, u0, tspan, p)
sol_ode_op = solve(prob_ode_op, Tsit5(), dt = δt)
plot(sol_ode_op, labels=["S" "I" "R"], lw=2, xlabel="Time", ylabel="Number")


@pyexec """
def sir_ode_ip_py(du,u,p,t):
    S = u[0]
    I = u[1]
    R = u[2]
    N = S+I+R
    beta = p[0]
    c = p[1]
    gamma = p[2]
    du[0] = dS = -beta*c*I/N*S
    du[1] = dI = beta*c*I/N*S - gamma*I
    du[2] = dR = gamma*I
""" => sir_ode_ip_py;


prob_ode_ip = ODEProblem{true}(sir_ode_ip_py, u0, tspan, p)
sol_ode_ip = solve(prob_ode_ip, Tsit5(), dt = δt)
plot(sol_ode_ip, labels=["S" "I" "R"], lw=2, xlabel="Time", ylabel="Number")


@benchmark solve(prob_ode_op, Tsit5(), dt = δt)


@benchmark solve(prob_ode_ip, Tsit5(), dt = δt)


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
prob_ode_julia = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode_julia = solve(prob_ode_julia, Tsit5(), dt = δt)
@benchmark solve(prob_ode_julia, Tsit5(), dt = δt)


using ModelingToolkit
@named sys_mtk = modelingtoolkitize(prob_ode_op);
prob_mtk = ODEProblem(sys_mtk, u0, tspan, p)
sol_mtk = solve(prob_mtk, Tsit5(), dt = δt)
plot(sol_mtk, labels=["S" "I" "R"], lw=2, xlabel="Time", ylabel="Number")


@benchmark solve(prob_mtk, Tsit5(), dt = $δt)

