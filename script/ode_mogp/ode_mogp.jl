
using OrdinaryDiffEq
using DiffEqCallbacks
using Surrogates
using Conda
using PyCall
using Random
using Plots
using BenchmarkTools;


env = Conda.ROOTENV
Conda.pip_interop(true, env)
Conda.pip("install", "mogp-emulator");


random = pyimport("random")
np = pyimport("numpy")
mogp = pyimport("mogp_emulator");


Random.seed!(123)
random.seed(123)
np.random.seed(123);


function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,γ) = p
    @inbounds begin
        du[1] = -β*S*I
        du[2] = β*S*I - γ*I
        du[3] = γ*I
    end
    nothing
end;


tmax = 10000.0
tspan = (0.0,tmax)
δt = 1.0;


n_train = 50 # Number of training samples
n_test = 1000; # Number of test samples


# Parameters are β, γ
lb = [0.00005, 0.1]
ub = [0.001, 1.0];


N = 1000.0
u0 = [990.0,10.0,0.0]
p = [0.0005,0.25]
prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


sampler = LatinHypercubeSample();


θ = Surrogates.sample(n_train,lb,ub,sampler);


logit = (x) -> log(x/(1-x))
invlogit = (x) -> exp(x)/(exp(x)+1.0)
cb_ss = TerminateSteadyState()
logit_final_size = function(z)
  prob = remake(prob_ode;p=z)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  fsp = sol[end][3]/N
  logit(fsp)
end;


lfs = logit_final_size.(θ);


gp = mogp.GaussianProcess(θ, lfs, nugget="fit");


gp = mogp.fit_GP_MAP(gp, n_tries=100);


lfs_train_pred = gp.predict(θ);


scatter(invlogit.(lfs),
        invlogit.(lfs_train_pred["mean"]),
        xlabel = "Model final size",
        ylabel = "Surrogate final size",
        legend = false,
        title = "Training set")


θ_test = sample(n_test,lb,ub,sampler)
lfs_test = logit_final_size.(θ_test)
lfs_test_pred = gp.predict(θ_test);


scatter(invlogit.(lfs_test),
        invlogit.(lfs_test_pred["mean"]),
        xlabel = "Model final size",
        ylabel = "Surrogate final size",
        legend = false,
        title = "Test set")


β_grid = collect(lb[1]:0.00001:ub[1])
θ_eval = [[βᵢ,0.25] for βᵢ in β_grid]
lfs_eval = gp.predict(θ_eval)
fs_eval = invlogit.(lfs_eval["mean"])
fs_eval_uc = invlogit.(lfs_eval["mean"] .+ 1.96 .* sqrt.(lfs_eval["unc"]))
fs_eval_lc = invlogit.(lfs_eval["mean"] .- 1.96 .* sqrt.(lfs_eval["unc"]))
plot(β_grid,
     fs_eval,
     xlabel = "Infectivity parameter, β",
     ylabel = "Final size",
     label = "Model")
plot!(β_grid,
      invlogit.(logit_final_size.(θ_eval)),
      ribbon = (fs_eval .- fs_eval_lc, fs_eval_uc - fs_eval),
      label = "Surrogate",
      legend = :right)


γ_grid = collect(lb[2]:0.001:ub[2])
θ_eval = [[0.001,γᵢ] for γᵢ in γ_grid]
lfs_eval = gp.predict(θ_eval)
fs_eval = invlogit.(lfs_eval["mean"])
fs_eval_uc = invlogit.(lfs_eval["mean"] .+ 1.96 .* sqrt.(lfs_eval["unc"]))
fs_eval_lc = invlogit.(lfs_eval["mean"] .- 1.96 .* sqrt.(lfs_eval["unc"]))
plot(γ_grid,
     fs_eval,
     xlabel = "Recovery rate, γ",
     ylabel = "Final size",
     label = "Model")
plot!(γ_grid,
      invlogit.(logit_final_size.(θ_eval)),
      ribbon = (fs_eval .- fs_eval_lc, fs_eval_uc - fs_eval),
      label = "Surrogate")


obs = logit_final_size(p)
invlogit(obs)


hm = mogp.HistoryMatching(gp=gp,
                          obs=obs,
                          coords=np.array(θ_test),
                          threshold=3.0);


nroy_points = hm.get_NROY() .+ 1
length(nroy_points),n_test


x = [θᵢ[1] for θᵢ in θ_test]
y = [θᵢ[2] for θᵢ in θ_test]
l = @layout [a b]
pl1 = histogram(x[nroy_points],legend=false,xlim=(lb[1],ub[1]),bins=lb[1]:0.00005:ub[1],title="NROY values for β")
vline!(pl1,[p[1]])
pl2 = histogram(y[nroy_points],legend=false,xlim=(lb[2],ub[2]),bins=lb[2]:0.05:ub[2],title="NROY values for γ")
vline!(pl2,[p[2]])
plot(pl1, pl2, layout = l)


@benchmark logit_final_size(p)


@benchmark gp.predict(p)

