
using OrdinaryDiffEq
using RCall
using Plots
using BenchmarkTools


R"""
sir_ode_odin <- odin::odin({
    ## Derivatives
    deriv(S) <- -beta*cee*S*I/N
    deriv(I) <- beta*cee*S*I/N-gamma*I
    deriv(R) <- gamma*I
    N <- S + I + R
    
    ## Initial conditions
    u[] <- user()
    dim(u) <- 3
    initial(S) <- u[1]
    initial(I) <- u[2]
    initial(R) <- u[3]
  
    ## Parameters
    p[] <- user()
    dim(p) <- 3
    beta <- p[1]
    cee <- p[2]
    gamma <- p[3]
  }, verbose=FALSE, target="c")
""";


R"""
sir_ode_odin_model <- sir_ode_odin$new(user=list(u=c(990.0,10.0,0.0),
                                       p=c(0.05,10.0,0.25)))

sir_ode_odin_f <- function(u,p,t){
    sir_ode_odin_model$set_user(user=list(u=u,p=p))
    return(sir_ode_odin_model$deriv(t,u))
}
""";


function sir_ode_odin_jl(u,p,t)
    robj = rcall(:sir_ode_odin_f, u, p, t)
    return convert(Array,robj)
end;


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode_odin = ODEProblem{false}(sir_ode_odin_jl, u0, tspan, p)
sol_ode_odin = solve(prob_ode_odin, Tsit5(), dt = δt)
plot(sol_ode_odin, labels=["S" "I" "R"], lw=2, xlabel="Time", ylabel="Number")


@benchmark solve(prob_ode_odin, Tsit5(), dt = δt)


function sir_ode_julia!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    du[1] = dS = -β*c*I/N*S
    du[2] = dI = β*c*I/N*S - γ*I
    du[3] = dR = γ*I
end
prob_ode_julia = ODEProblem(sir_ode_julia!, u0, tspan, p)
sol_ode_julia = solve(prob_ode_julia, Tsit5(), dt = δt)
@benchmark solve(prob_ode_julia, Tsit5(), dt = δt)


R"""
sir_ode_model <- sir_ode_odin$new(user=list(u=c(990.0,10.0,0.0),
                                       p=c(0.05,10.0,0.25)))
sir_ode_run <- function(t){
    return(sir_ode_model$run(t))
}
""";


out = rcall(:sir_ode_run, collect(0:δt:tmax))
@benchmark rcall(:sir_ode_run, collect(0:δt:tmax))

