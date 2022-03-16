
using OrdinaryDiffEq
using DiffEqCallbacks
using Optim
using Random
using Distributions
using StatsBase
using Plots
using BenchmarkTools


function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*S*I
        du[2] = β*S*I - γ*I
        du[3] = γ*I
    end
    nothing
end;


tmax = 10000.0
tspan = (0.0,tmax);


cb_ss = TerminateSteadyState();


N = 1000.0;
u0 = [990.0,10.0,0.0];
p = [0.0005,0.25]; # β,γ


prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


sol_ode = solve(prob_ode,Tsit5())
plot(sol_ode(0:1:40.0),
     xlabel="Time",
     ylabel="Number",
     labels=["S" "I" "R"])


lb = 0.00005
ub = 0.001
δ = 0.0000001;


θ = lb:δ:ub;


M = function(θ)
  p = prob_ode.p
  p[1] = θ
  prob = remake(prob_ode;p=p)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  ϕ = sol[end][3]/N
  ϕ
end;


using Roots
M_analytic = function(θ)
  _,γ = prob_ode.p
  β = θ
  R₀ = β*N/γ
  S0 = prob_ode.u0[1]/N
  f(ϕ) = 1-S0*exp(-R₀*ϕ)-ϕ
  ϕ = find_zero(f,(0.0,1.0))
  ϕ
end;


@benchmark M(p[1])


@benchmark M_analytic(p[1])


ϕ = M.(θ);


R₀ = θ*N/p[2];


function twiny(sp::Plots.Subplot)
    sp[:top_margin] = max(sp[:top_margin], 30Plots.px)
    plot!(sp.plt, inset = (sp[:subplot_index], bbox(0,0,1,1)))
    twinsp = sp.plt.subplots[end]
    twinsp[:xaxis][:mirror] = true
    twinsp[:background_color_inside] = RGBA{Float64}(0,0,0,0)
    Plots.link_axes!(sp[:yaxis], twinsp[:yaxis])
    twinsp
end
twiny(plt::Plots.Plot = current()) = twiny(plt[1]);


plot(θ, ϕ,
     xlabel="Infectivity parameter, β",
     ylabel="Final size, ϕ",
     legend=false)
pl = twiny()
plot!(pl, R₀, ϕ, xlabel = "Reproductive number, R₀", legend = false)


using KernelDensity
k_kd = kde_lscv(ϕ;boundary=(0.0,1.0),kernel=Normal)
q1star_kd = [pdf(k_kd, ϕᵢ) for ϕᵢ in ϕ];


using KernelDensityEstimate
k_kde = KernelDensityEstimate.kde!(ϕ)
q1star_kde = evaluateDualTree(k_kde,ϕ);


using AverageShiftedHistograms
k_ash = ash(ϕ)
q1star_ash = [AverageShiftedHistograms.pdf(k_ash,ϕᵢ) for ϕᵢ in ϕ];


using MultiKDE
k_mkde = KDEUniv(ContinuousDim(), 0.01, ϕ, MultiKDE.gaussian)
q1star_mkde = [MultiKDE.pdf(k_mkde, ϕᵢ, keep_all=false) for ϕᵢ in ϕ];


plot(ϕ,q1star_kd,label="KernelDensity",xlabel="Final size",ylabel="Density")
plot!(ϕ,q1star_kde,label="KernelDensityEstimate")
plot!(ϕ,q1star_ash,label="AverageShiftedHistograms")
plot!(ϕ,q1star_mkde,label="MultiKDE",legend=:top)


α = 0.5
q₂ = Distributions.pdf.(Uniform(0.05,0.1),ϕ)
L₁(θᵢ) = 1.0
L₂(ϕᵢ) = 1.0;


plot(ϕ,q₂,label="Prior on outputs, q₂")
plot!(ϕ,q1star_kd,label="Induced prior on outputs, q₁*")


w_kd = (q₂ ./ q1star_kd).^(1-α) .* L₁.(θ) .* L₂.(ϕ);


lw_kd = (1-α) .* log.(q₂) .- (1-α) .* log.(q1star_kd) .+ log.(L₁.(θ)) .+ log.(L₂.(ϕ));


w_kde = (q₂ ./ q1star_kde).^(1-α)
w_ash = (q₂ ./ q1star_ash).^(1-α)
w_mkde = (q₂ ./ q1star_mkde).^(1-α);


Random.seed!(123)
n = 1000
πθ = StatsBase.sample(θ, Weights(w_kd),n,replace=true);


mean(πθ),std(πθ)


histogram(πθ,
          bins=25,
          legend=false,
          xlim=(lb,ub),
          xlabel="Infectivity parameter, β",
          ylabel="Count")


w_norm = w_kd ./ sum(w_kd)
ess = 1.0/sum(w_norm .^ 2)


using ThreadsX
θ_fine = lb:1e-10:ub
# We use ThreadsX instead of ϕ_fine = M_analytic.(θ_fine)
ϕ_fine = ThreadsX.collect(M_analytic(θᵢ) for θᵢ in θ_fine)
k_kd_fine = kde_lscv(ϕ_fine;boundary=(0.0,1.0),kernel=Normal)
#k_kde_fine = KernelDensityEstimate.kde!(ϕ_fine)
k_ash_fine = ash(ϕ_fine)
k_mkde_fine = KDEUniv(ContinuousDim(), 0.01, ϕ_fine, MultiKDE.gaussian)
q1star_kd_fine = [pdf(k_kd_fine, ϕᵢ) for ϕᵢ in ϕ]
#q1star_kde_fine = evaluateDualTree(k_kde_fine,ϕ_fine)
q1star_ash_fine = [AverageShiftedHistograms.pdf(k_ash_fine,ϕᵢ) for ϕᵢ in ϕ]
q1star_mkde_fine = ThreadsX.collect(MultiKDE.pdf(k_mkde_fine, ϕᵢ, keep_all=false) for ϕᵢ in ϕ);


l = @layout [a; b; c]
pl1 = plot(ϕ,q1star_kd_fine,label="KernelDensity fine",xlabel="Final size",ylabel="Density")
plot!(pl1,ϕ,q1star_kd,label="KernelDensity coarse")
pl2=plot(ϕ,q1star_ash_fine,label="AverageShiftedHistograms fine")
plot!(pl2,ϕ,q1star_ash,label="AverageShiftedHistograms coarse")
pl3=plot(ϕ,q1star_mkde,label="MultiKDE")
plot!(pl3,ϕ,q1star_mkde_fine,label="MultiKDE fine")
plot(pl1, pl2, pl3, layout=l)

