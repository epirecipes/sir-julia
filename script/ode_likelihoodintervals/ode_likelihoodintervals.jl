
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


lb = [0.001, 0.01] # Lower bound
ub = [0.1, 0.1] # Upper bound
θ = [0.01, 0.05] # Exact values
θ₀ = [0.002, 0.08]; # Initial conditions for optimization


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


crit_val = 0.5*quantile(Chisq(2),0.95)


regular_grid = RegularGrid(lb, ub, 10)


gs, loglik_vals = grid_search(prob, regular_grid; save_vals=Val(true), parallel = Val(true))
gs


gs_max_lik, gs_max_idx = findmax(loglik_vals);


nroy = loglik_vals .>= (gs_max_lik - crit_val)
nroyp = [ProfileLikelihood.get_parameters(regular_grid,(i,j)) for i in 1:100 for j in 1:100 if nroy[i,j]==1]


lb2 = [minimum([x for x in nroyp[i]]) for i in 1:2] .* 0.5
ub2 = [maximum([x for x in nroyp[i]]) for i in 1:2] .* 2


n_samples = 10000
parameter_vals = QuasiMonteCarlo.sample(n_samples, lb2, ub2, LatinHypercubeSample());


irregular_grid = IrregularGrid(lb, ub, parameter_vals);


gs_ir, loglik_vals_ir = grid_search(prob, irregular_grid; save_vals=Val(true), parallel = Val(true))
gs_ir


prob = update_initial_estimate(prob, gs_ir)
sol = mle(prob, Optim.LBFGS())
θ̂ = get_mle(sol)


fig = Figure(fontsize=38)
i₀_grid = get_range(regular_grid, 1)
#i₀_grid_ir = [ProfileLikelihood.get_parameters(irregular_grid,i)[1] for i in 1:10000]
β_grid = get_range(regular_grid, 2)
#β_grid_ir = [ProfileLikelihood.get_parameters(irregular_grid,i)[2] for i in 1:10000]
ax = Axis(fig[1, 1],
    xlabel=L"i_0", ylabel=L"\beta")
co = heatmap!(ax, i₀_grid, β_grid, loglik_vals, colormap=Reverse(:matter))
contour!(ax, i₀_grid, β_grid, loglik_vals, levels=40, color=:black, linewidth=1 / 4)
contour!(ax, i₀_grid, β_grid, loglik_vals, levels=[minimum(loglik_vals), maximum(loglik_vals)-crit_val], color=:red, linewidth=1 / 2)
scatter!(ax, [θ[1]], [θ[2]], color=:blue, markersize=14)
scatter!(ax, [θ̂[1]], [θ̂[2]], color=:red, markersize=14)
clb = Colorbar(fig[1, 2], co, label=L"\ell(i_0, \beta)", vertical=true)
fig


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
end


npts = 1000
t_pred = LinRange(tspan[1]+1, tspan[2], npts)
d = Dict("tspan" => tspan, "npts" => npts);


exact_soln = prediction_function(θ, d)
mle_soln = prediction_function(θ̂, d)


threshold = maximum(loglik_vals_ir)-crit_val
θₗₕₛ = [ProfileLikelihood.get_parameters(irregular_grid,i) for i in 1:10000 if loglik_vals_ir[i] >= threshold]
lhs_soln = [prediction_function(theta,d) for theta in θₗₕₛ]


lhs_lci = vec(minimum(hcat(pred_lhs...),dims=2))
lhs_uci = vec(maximum(hcat(pred_lhs...),dims=2));


fig = Figure(fontsize=32, resolution=(500, 400))
ax = Axis(fig[1, 1], width=400, height=300)
lines!(ax, t_pred, lci, color=:black, linewidth=3)
lines!(ax, t_pred, uci, color=:black, linewidth=3)
band!(ax, t_pred, lci, uci, color=:grey)
lines!(ax, t_pred, exact_soln, color=:red)
lines!(ax, t_pred, mle_soln, color=:blue, linestyle=:dash)
fig

