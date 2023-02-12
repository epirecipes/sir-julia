
using ModelingToolkit
using OrdinaryDiffEq
using DataFrames
using StatsPlots
using LaTeXStrings;


@parameters t R₀ γ μ a
@variables S(t) I(t) β(t)
D = Differential(t)

eqs = [D(S) ~ μ - β*S*I - μ*S,
       D(I) ~ β*S*I - (γ+μ)*I,
       β ~ R₀*(γ+μ)*(1+a*cos(2*π*t))];


S₀ = 1/R₀
I₀ = (μ/(μ+γ))*(1-S₀)
u₀ = [S => S₀, I => I₀]
p = [μ => 0.02, γ => 28.08, R₀ => 17.0, a => 0.08];


tmin = 0.0
tmax = 650
transient = 600
strobe = 1.0
tspan = (tmin, tmax);


solver = RK4()
tol = 1e-11
maxiters = 1e7;


@named sys = ODESystem(eqs)
simpsys = structural_simplify(sys)
prob = ODEProblem(simpsys, u₀, tspan, p; jac=true);


R0vec = collect(1.0:0.01:30.0);


function prob_func(prob, i, repeat)
    return remake(prob, p=[μ => 0.02, γ => 28.08, R₀ => R0vec[i], a => 0.08])
end;


function output_func(sol, i)
    strobetimes = collect(transient:strobe:tmax)
    df = DataFrame(sol(strobetimes))
    rename!(df,[:t, :S, :I])
    df[!,"R0"] = fill(R0vec[i],length(strobetimes))
    df[!,"LogS"] = log10.(abs.(df.S))
    df[!,"LogI"] = log10.(abs.(df.I))
    return (df, false)
end;


ensemble_prob = EnsembleProblem(prob,
                                prob_func = prob_func,
                                output_func = output_func)

@time sim = solve(ensemble_prob,
                  solver,
                  EnsembleThreads(),
                  trajectories = length(R0vec);
                  maxiters = maxiters,
                  isoutofdomain=(u,p,t) -> any(x -> x <= 0 || x >= 1, u),
                  abstol=tol);


results = vcat(sim...);


@df results scatter(:R0,
                    :LogI,
                    xlabel=L"R_0",
                    ylabel=L"log_{10} I",
                    markersize=1.0,
                    color=:gray,
                    legend=false,
                    ylims=(-6,-2))


times = collect(transient:0.01:transient+10.0);


function plot_solution_R₀(R0, prob=prob)
    prob = remake(prob, p=[μ => 0.02, γ => 28.08, R₀ => R0, a => 0.08])
    sol = solve(prob,
              solver;
              maxiters = maxiters,
              isoutofdomain=(u,p,t) -> any(x -> x <= 0 || x >= 1, u),
              abstol=tol);
    # Note how we can specify transformations of the output variable
    results = DataFrame(sol(times, idxs=log10(I)))
    rename!(results, [:t, :Log10I])
    plt = @df results plot(:t,
                   :Log10I,
                   xlabel=L"t",
                   ylabel=L"log_{10} I",
                   title=L"R_0=%$(R0)",
                   legend=false,
                   xticks=collect(600:2:610))
    return plt
end;


R0plotvec = [15.0, 16.0, 17.0, 18.0]
plts = [plot_solution_R₀(R0) for R0 in R0plotvec]
l = @layout [a b; c d]
plot(plts..., layout=l)

