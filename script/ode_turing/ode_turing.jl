
using Pkg
Pkg.instantiate()


using OrdinaryDiffEq
using SciMLSensitivity
using Random
using Distributions
using Turing
using DataFrames
using StatsPlots
using BenchmarkTools


function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end;


tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax
u0 = [990.0,10.0,0.0,0.0] # S,I.R,C
p = [0.05,10.0,0.25]; # β,c,γ


prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0);


C = Array(sol_ode)[4,:] # Cumulative cases
X = C[2:end] - C[1:(end-1)];


Random.seed!(1234)
Y = rand.(Poisson.(X));


bar(obstimes,Y,legend=false)
plot!(obstimes,X,legend=false)


@model function bayes_sir(y)
  # Calculate number of timepoints
  l = length(y)
  i₀  ~ Uniform(0.0,1.0)
  β ~ Uniform(0.0,1.0)
  I = i₀*1000.0
  u0=[1000.0-I,I,0.0,0.0]
  p=[β,10.0,0.25]
  tspan = (0.0,float(l))
  prob = ODEProblem(sir_ode!,
          u0,
          tspan,
          p)
  sol = solve(prob,
              Tsit5(),
              saveat = 1.0)
  sol_C = Array(sol)[4,:] # Cumulative cases
  sol_X = sol_C[2:end] - sol_C[1:(end-1)]
  l = length(y)
  for i in 1:l
    y[i] ~ Poisson(abs(sol_X[i]))
  end
end;


@time ode_nuts = sample(bayes_sir(Y), NUTS(0.65), 10000, verbose=false, progress=false);


describe(ode_nuts)


plot(ode_nuts)


posterior = DataFrame(ode_nuts);


histogram2d(posterior[!,:β],posterior[!,:i₀],
                bins=80,
                xlabel="β",
                ylab="i₀",
                ylim=[0.006,0.016],
                xlim=[0.045,0.055],
                legend=false)
plot!([0.05,0.05],[0.0,0.01])
plot!([0.0,0.05],[0.01,0.01])


function predict(y,chain)
    # Length of data
    l = length(y)
    # Length of chain
    m = length(chain)
    # Choose random
    idx = sample(1:m)
    i₀ = chain[:i₀][idx]
    β = chain[:β][idx]
    I = i₀*1000.0
    u0=[1000.0-I,I,0.0,0.0]
    p=[β,10.0,0.25]
    tspan = (0.0,float(l))
    prob = ODEProblem(sir_ode!,
            u0,
            tspan,
            p)
    sol = solve(prob,
                Tsit5(),
                saveat = 1.0)
    out = Array(sol)
    sol_X = [0.0; out[4,2:end] - out[4,1:(end-1)]]
    hcat(sol_ode.t,out',sol_X)
end;


Xp = []
for i in 1:10
    pred = predict(Y,ode_nuts)
    push!(Xp,pred[2:end,6])
end;


scatter(obstimes,Y,legend=false)
plot!(obstimes,Xp,legend=false)


@benchmark sample(bayes_sir(Y), NUTS(0.65), 10000, verbose=false, progress=false)


using Base.Threads


Threads.nthreads()


function sir_ode_solve(problem, l, i₀, β)
    I = i₀*1000.0
    S = 1000.0 - I
    u0 = [S, I, 0.0, 0.0]
    p = [β, 10.0, 0.25]
    prob = remake(problem; u0=u0, p=p)
    sol = solve(prob, Tsit5(), saveat = 1.0)
    sol_C = view(sol, 4, :) # Cumulative cases
    sol_X = Array{eltype(sol_C)}(undef, l)
    @inbounds @simd for i in 1:length(sol_X)
        sol_X[i] = sol_C[i + 1] - sol_C[i]
    end
    return sol_X
end;


function simulate_data(l, i₀, β)
    prob = ODEProblem(sir_ode!, [990.0, 10.0, 0.0, 0.0], (0.0, l), [β, 10.0, 0.25])
    X = sir_ode_solve(prob, l, i₀, β)
    Y = rand.(Poisson.(X))
    return X, Y
end;


nsims = 1000
i₀_true = 0.01
β_true = 0.05
l = 40
i₀_mean = Array{Float64}(undef, nsims)
β_mean = Array{Float64}(undef, nsims)
i₀_coverage = Array{Float64}(undef, nsims)
β_coverage = Array{Float64}(undef, nsims)
Threads.@threads for i in 1:nsims
    X_sim, Y_sim = simulate_data(l, i₀_true, β_true)
    r = sample(bayes_sir(Y_sim), NUTS(0.65), 10000, verbose=false, progress=false)
    i₀_mean[i] = mean(r[:i₀])
    i0_cov = sum(r[:i₀] .<= i₀_true) / length(r[:i₀])
    β_mean[i] = mean(r[:β])
    b_cov = sum(r[:β] .<= β_true) / length(r[:β])
    i₀_coverage[i] = i0_cov
    β_coverage[i] = b_cov
end;


# Convenience function to check if the true value is within the credible interval
function in_credible_interval(x, lwr=0.025, upr=0.975)
    return x >= lwr && x <= upr
end;


pl_β_coverage = histogram(β_coverage, bins=0:0.1:1.0, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
pl_i₀_coverage = histogram(i₀_coverage, bins=0:0.1:1.0, label=false, title="i₀", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
plot(pl_β_coverage, pl_i₀_coverage, layout=(1,2), plot_title="Distribution of CDF of true value")


sum(in_credible_interval.(β_coverage)) / nsims


sum(in_credible_interval.(i₀_coverage)) / nsims


pl_β_mean = histogram(β_mean, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.045, 0.055))
vline!([β_true], label="True value")
pl_i₀_mean = histogram(i₀_mean, label=false, title="i₀", ylabel="Density", density=true, xrotation=45, xlim=(0.0,0.02))
vline!([i₀_true], label="True value")
plot(pl_β_mean, pl_i₀_mean, layout=(1,2), plot_title="Distribution of posterior means")

