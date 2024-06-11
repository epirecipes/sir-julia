
using OrdinaryDiffEq
using RCall
using Plots
using BenchmarkTools


R"""
sir_ode_op_r <- function(u,p,t){
    S <- u[1]
    I <- u[2]
    R <- u[3]
    N <- S+I+R
    beta <- p[1]
    cee <- p[2]
    gamma <- p[3]
    dS <- -beta*cee*I/N*S
    dI <- beta*cee*I/N*S - gamma*I
    dR <- gamma*I
    return(c(dS,dI,dR))
}
""";


function sir_ode_op_jl(u,p,t)
    robj = rcall(:sir_ode_op_r, u, p, t)
    return convert(Array,robj)
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode_op = ODEProblem{false}(sir_ode_op_jl, u0, tspan, p)
sol_ode_op = solve(prob_ode_op, Tsit5(), dt = δt)
plot(sol_ode_op, labels=["S" "I" "R"], lw = 2, xlabel = "Time", ylabel = "Number")


@benchmark solve(prob_ode_op, Tsit5(), dt = δt)


function sir_ode_op_julia(u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    dS = -β*c*I/N*S
    dI = β*c*I/N*S - γ*I
    dR = γ*I
    [dS,dI,dR]
end
prob_ode_julia = ODEProblem(sir_ode_op_julia, u0, tspan, p)
sol_ode_julia = solve(prob_ode_julia, Tsit5(), dt = δt)
@benchmark solve(prob_ode_julia, Tsit5(), dt = δt)

