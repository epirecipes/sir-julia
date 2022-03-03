
using OrdinaryDiffEq
using Distributions
using DiffEqUncertainty
using Plots


function my_centralmoment(n::Int, g::Function, args...; kwargs...)
    if n < 2 return Float64[] end
    # Compute the expectations of g, g^2, ..., g^n
    sol = expectation(x -> [g(x)^i for i in 1:n], args...; nout = n, kwargs...)
    exp_set = sol[:]
    mu_g = popfirst!(exp_set)
    # Combine according to binomial expansion
    const_term(n) = (-1)^(n-1) * (n-1) * mu_g^n
    binom_term(n, k, mu, exp_gi) = binomial(n, k) * (-mu)^(n - k) * exp_gi
    binom_sum = function (exp_vals)
        m = length(exp_vals) + 1
        sum([binom_term(m, k + 1, mu_g, v) for (k,v) in enumerate(exp_vals)]) + const_term(m)
    end
    return [zero(exp_set[1]), [binom_sum(exp_set[1:i]) for i in 1:length(exp_set)]...]
end


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
end;


δt = 1.0
tmax = 40.0
tspan = (0.0,tmax);
t = 0:δt:tmax;


u0 = [990.0,10.0,0.0]
p = [0.05,10,0.25]
prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


p_dist = [Uniform(0.01,0.1),
     Uniform(5,20.0),
     Uniform(0.1,1.0)];


g(sol) = sol(t)
g(sol,x,i) = sol(x)[i];


n_samples = 1000
sol_ode_mean_mc = expectation(g, prob_ode, u0, p_dist, MonteCarlo(), Tsit5(); trajectories = n_samples)
sol_ode_mean_mc = Array(sol_ode_mean_mc)'


sol_ode_std_mc = [[sqrt(my_centralmoment(2, (sol) -> g(sol,x,i), prob_ode, u0, p_dist, MonteCarlo(), Tsit5(); trajectories = n_samples)[2]) for x in t] for i in 1:3]
sol_ode_std_mc = hcat(sol_ode_std_mc...)


sol_ode_mean_k = expectation(g, prob_ode, u0, p_dist, Koopman(), Tsit5())
sol_ode_mean_k = Array(sol_ode_mean_k)'


sol_ode_std_k = [[sqrt(my_centralmoment(2, (sol) -> g(sol,x,i), prob_ode, u0, p_dist, Koopman(), Tsit5())[2]) for x in t] for i in 1:3]
sol_ode_std_k = hcat(sol_ode_std_k...)


l = @layout [a b]
pl1 = plot(t,
     sol_ode_mean_mc,
     ribbon=sol_ode_std_mc,
     fillalpha=0.15,
     label=["S" "I" "R"],
     xlabel="Time",
     ylabel="Number",
     title="Monte Carlo")
pl2 = plot(t,
     sol_ode_mean_k,
     ribbon=sol_ode_std_k,
     fillalpha=0.15,
     label=["S" "I" "R"],
     xlabel="Time",
     ylabel="Number",
     title="Koopman")
plot(pl1,pl2,layout=l)

