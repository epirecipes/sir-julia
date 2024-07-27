
using Pkg
Pkg.instantiate()


using RxInfer
using OrdinaryDiffEq
using ExponentialFamilyProjection
using Optimisers
using StableRNGs
using StatsFuns
using StaticArrays
using Plots
using BenchmarkTools
import BayesBase;


function sir_ode(u, p, t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    return @SArray([
         -infection,
        infection - recovery,
        recovery,
        infection
    ]) # S, I, R, C
end;


function sir_ode_solve(problem, l, i₀, β)
    I = i₀*1000.0
    S = 1000.0 - I
    u0 = @SArray([S, I, 0.0, 0.0])
    p = @SArray([β, 10.0, 0.25])
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
    prob = ODEProblem(sir_ode, @SArray([990.0, 10.0, 0.0, 0.0]), (0.0, l), @SArray([β, 10.0, 0.25]))
    X = sir_ode_solve(prob, l, i₀, β)
    Y = rand.(Poisson.(X))
    return X, Y
end;


struct ODEFused{I, B, L, F} <: DiscreteMultivariateDistribution
    i₀::I
    β::B
    l::L
    problem::F
end;


function BayesBase.logpdf(ode::ODEFused, y::Vector)
    sol_X = sir_ode_solve(ode.problem, ode.l, ode.i₀, ode.β)
    # `sum` over individual entries of the result of the `ODE` solver
    sumlpdf = sum(zip(sol_X, y)) do (x_i, y_i)
        return logpdf(Poisson(abs(x_i)), y_i)
    end
    # `clamp` to avoid infinities in the beginning, where 
    # priors are completely off
    return clamp(sumlpdf, -100000, Inf)
end;


function BayesBase.insupport(ode::ODEFused, y::Vector)
    return true
end

function BayesBase.mean(p::PointMass{D}) where { D <: ODEProblem }
    return p.point
end;


@node ODEFused Stochastic [ y, i₀, β, l, problem ];


@model function bayes_sir(y)
    l = length(y)
    prob = ODEProblem(sir_ode, @SArray([990.0, 10.0, 0.0, 0.0]), (0.0, l), @SArray([0.05, 10.0, 0.25]))    
    i₀ ~ Beta(1.0, 1.0)
    β  ~ Beta(1.0, 1.0)
    y  ~ ODEFused(i₀, β, l, prob)
end;


@constraints function sir_constraints()
    parameters = ProjectionParameters(
        strategy = ExponentialFamilyProjection.ControlVariateStrategy(nsamples = 200)
    )

    # In principle different parameters can be used for different states
    q(i₀) :: ProjectedTo(Beta; parameters = parameters)
    q(β) :: ProjectedTo(Beta; parameters = parameters)

    # `MeanField` is required for `NodeFunctionRuleFallback`
    q(i₀, β) = MeanField()
end;


@initialization function sir_initialization()
    q(β)  = Beta(1, 1)
    q(i₀) = Beta(1, 1)
end;


β_true = 0.05
i₀_true = 0.01
l = 40
X, Y = simulate_data(l, i₀_true, β_true);


ts = 1.0:1.0:l
plot(ts, X, label="Deterministic mean", xlabel="Time", ylabel="Daily incidence")
scatter!(ts, Y, label="Simulated observations")


niter = 15
result = infer(
        model = bayes_sir(),
        data  = (y = Y, ),
        constraints = sir_constraints(),
        initialization = sir_initialization(),
        iterations = niter,
        showprogress = false,
        options = (
            # Read `https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/undefinedrules/`
            rulefallback = NodeFunctionRuleFallback(),
        )
);


pl_β_mean_i = plot(1:15, [mean(x) for x in result.posteriors[:β]], label=false, xlabel="Iteration", ylabel="Mean", title="β")
pl_i₀_mean_i = plot(1:15, [mean(x) for x in result.posteriors[:i₀]], label=false, xlabel="Iteration", ylabel="Mean", title="i₀")
plot(pl_β_mean_i, pl_i₀_mean_i, layout=(1,2), plot_title="Mean of posterior by iteration")


posterior_i₀ = result.posteriors[:i₀][end]
posterior_β = result.posteriors[:β][end];


mean_var(posterior_i₀) # Should be 0.01


mean_var(posterior_β) # Should be 0.05


p1 = plot(0.0:0.0001:0.02, x -> pdf(posterior_i₀, x); label="q(i₀)")
vline!(p1, [i₀_true], label=false)
p2 = plot(0.04:0.0001:0.06, x -> pdf(posterior_β, x); label="q(β)")
vline!(p2, [β_true], label=false)
plot(p1, p2)


@benchmark infer(
        model = bayes_sir(),
        data  = (y = Y, ),
        constraints = sir_constraints(),
        initialization = sir_initialization(),
        iterations = niter,
        showprogress = false,
        options = (
            # Read `https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/undefinedrules/`
            rulefallback = NodeFunctionRuleFallback(),
        )
)


using Base.Threads


Threads.nthreads()


nsims = 1000
i₀_mean = Array{Float64}(undef, nsims)
β_mean = Array{Float64}(undef, nsims)
i₀_coverage = Array{Float64}(undef, nsims)
β_coverage = Array{Float64}(undef, nsims)
Threads.@threads for i in 1:nsims
    X_sim, Y_sim = simulate_data(l, i₀_true, β_true)
    r = infer(
              model = bayes_sir(),
              data  = (y = Y_sim, ),
              constraints = sir_constraints(),
              initialization = sir_initialization(),
              iterations = niter,
              showprogress = false,
              options = ( rulefallback = NodeFunctionRuleFallback(), ))
    i0 = r.posteriors[:i₀][end]
    i₀_mean[i] = mean(i0)
    i0_cov = cdf(i0, i₀_true)
    b = r.posteriors[:β][end]
    β_mean[i] = mean(b)
    b_cov = cdf(b, β_true)
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

