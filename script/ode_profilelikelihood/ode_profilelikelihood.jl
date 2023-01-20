
using OrdinaryDiffEq
using ProfileLikelihood
using StatsFuns
using Random
using Distributions
using Optimization
using OptimizationOptimJL
using QuasiMonteCarlo
using CairoMakie
using LaTeXStrings
using DataFrames


function sir_ode!(du, u, p, t)
    (S, I, R, C) = u
    (β, c, γ) = p
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


δt = 1.0
tmax = 40.0
tspan = (0.0,tmax);


u₀ = [990.0, 10.0, 0.0, 0.0]; # S, I, R, C


p = [0.05,10.0,0.25]; # β, c, γ


prob_ode = ODEProblem(sir_ode!, u₀, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), saveat=δt);


out = Array(sol_ode);


colors = [:blue, :red, :green, :purple]
legends = ["S", "I", "R", "C"]
fig = Figure()
ax = Axis(fig[1, 1])
for i = 1:4
    lines!(ax, sol_ode.t, out[i,:], label = legends[i], color = colors[i])
end
axislegend(ax)
ax.xlabel = "Time"
ax.ylabel = "Number"
fig


C = out[4,:];
X = C[2:end] .- C[1:(end-1)];


Random.seed!(1234);
data = rand.(Poisson.(X));


function ll(θ, data, integrator)
    (i0,β) = θ
    integrator.p[1] = β
    integrator.p[2] = 10.0
    integrator.p[3] = 0.25
    I = i0*1000.0
    u₀=[1000.0-I,I,0.0,0.0]
    reinit!(integrator, u₀)
    solve!(integrator)
    sol = integrator.sol
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    sum(logpdf.(Poisson.(X),data))
end;


lb = [0.0, 0.0]
ub = [1.0, 1.0]
θ = [0.01, 0.05]
θ₀ = [0.01, 0.1];


integrator = init(prob_ode, Tsit5(); saveat = δt) # takes the same arguments as `solve`
ll(θ₀, data, integrator)


syms = [:i₀, :β]
prob = LikelihoodProblem(
    ll, θ₀, sir_ode!, u₀, tmax; 
    syms=syms,
    data=data,
    ode_parameters=p, # temp values for p
    ode_kwargs=(verbose=false, saveat=δt),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
);


sol = mle(prob, NelderMead())
θ̂ = get_mle(sol);


prof = profile(prob, sol; alg=NelderMead(), parallel=false)
confints = get_confidence_intervals(prof);


fig = plot_profiles(prof; latex_names=[L"i_0", L"\beta"])
fig


ENV["COLUMNS"]=80
df_res = DataFrame(
    Parameters = [:i₀, :β], 
    CILower = [confints[i][1] for i in 1:2],
    CIUpper = [confints[i][2] for i in 1:2],
    FittedValues = θ̂,
    TrueValues = [0.01,0.05],
    NominalStartValues = θ₀
)
df_res


function prediction_function(θ, data)
    (i0,β) = θ
    tspan = data["tspan"]
    npts = data["npts"]
    t2 = LinRange(tspan[1]+1, tspan[2], npts)
    t1 = LinRange(tspan[1], tspan[2]-1, npts)
    I = i0*1000.0
    prob = remake(prob_ode,u0=[1000.0-I,I,0.0,0.0],p=[β,10.0,0.25],tspan=tspan)
    sol = solve(prob,Tsit5())
    return sol(t2)[4,:] .- sol(t1)[4,:]
end;


npts = 1000
t_pred = LinRange(tspan[1]+1, tspan[2], npts)
d = Dict("tspan" => tspan, "npts" => npts);


exact_soln = prediction_function([0.01,0.05], d)
mle_soln = prediction_function(θ̂, d);


parameter_wise, union_intervals, all_curves, param_range =
    get_prediction_intervals(prediction_function,
                             prof,
                             d);


fig = Figure(fontsize=32, resolution=(1800, 900))
alp = join('a':'b')
latex_names = [L"i_0", L"\beta"]
for i in 1:2
    ax = Axis(fig[1, i], title=L"(%$(alp[i])): Profile-wise PI for %$(latex_names[i])",
        titlealign=:left, width=400, height=300)
    band!(ax, t_pred, getindex.(parameter_wise[i], 1), getindex.(parameter_wise[1], 2), color=(:grey, 0.7), transparency=true)
    lines!(ax, t_pred, exact_soln, color=:red)
    lines!(ax, t_pred, mle_soln, color=:blue, linestyle=:dash)
    lines!(ax, t_pred, getindex.(parameter_wise[i], 1), color=:black, linewidth=3)
    lines!(ax, t_pred, getindex.(parameter_wise[i], 2), color=:black, linewidth=3)
end
ax = Axis(fig[1,3], title=L"(c):$ $ Union of all intervals",
    titlealign=:left, width=400, height=300)
band!(ax, t_pred, getindex.(union_intervals, 1), getindex.(union_intervals, 2), color=(:grey, 0.7), transparency=true)
lines!(ax, t_pred, getindex.(union_intervals, 1), color=:black, linewidth=3)
lines!(ax, t_pred, getindex.(union_intervals, 2), color=:black, linewidth=3)
lines!(ax, t_pred, exact_soln, color=:red)
lines!(ax, t_pred, mle_soln, color=:blue, linestyle=:dash)
fig

