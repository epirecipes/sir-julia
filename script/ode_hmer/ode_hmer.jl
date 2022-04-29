
using OrdinaryDiffEq
using DiffEqCallbacks
using Surrogates
using DataFrames
using RCall
using Random
using Plots;


R"library(hmer)";


Random.seed!(123)
R"set.seed(123)";


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


training_df = DataFrame(θ)
rename!(training_df,["b","g"])
training_df[!,:lfs] = lfs;


input_ranges = Dict(:b => [lb[1], ub[1]], :g => [lb[2], ub[2]])
output_names = ["lfs"];


@rput training_df
@rput output_names
@rput input_ranges;


R"emulator <- emulator_from_data(training_df, output_names, input_ranges)"


θ_test = sample(n_test,lb,ub,sampler)
lfs_test = logit_final_size.(θ_test);


test_df = DataFrame(θ_test)
rename!(test_df,["b","g"])
test_df[!,:lfs] = lfs_test
@rput test_df; # copy to R


R"lfs_test_pred <- list()"
R"lfs_test_pred$mean <- emulator$lfs$get_exp(test_df)"
R"lfs_test_pred$unc <- emulator$lfs$get_cov(test_df)"
@rget lfs_test_pred;


scatter(invlogit.(lfs_test),
        invlogit.(lfs_test_pred[:mean]),
        xlabel = "Model final size",
        ylabel = "Surrogate final size",
        legend = false,
        title = "Test set")


β_grid = collect(lb[1]:0.00001:ub[1])
θ_eval = [(βᵢ,0.25) for βᵢ in β_grid]
eval_df = DataFrame(θ_eval)
rename!(eval_df,["b","g"])
@rput eval_df
R"lfs_eval <- list()"
R"lfs_eval$mean <- emulator$lfs$get_exp(eval_df)"
R"lfs_eval$unc <- emulator$lfs$get_cov(eval_df)"
@rget lfs_eval
fs_eval = invlogit.(lfs_eval[:mean])
fs_eval_uc = invlogit.(lfs_eval[:mean] .+ 1.96 .* sqrt.(lfs_eval[:unc]))
fs_eval_lc = invlogit.(lfs_eval[:mean] .- 1.96 .* sqrt.(lfs_eval[:unc]))
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
θ_eval = [(0.001,γᵢ) for γᵢ in γ_grid]
eval_df = DataFrame(θ_eval)
rename!(eval_df,["b","g"])
@rput eval_df
R"lfs_eval <- list()"
R"lfs_eval$mean <- emulator$lfs$get_exp(eval_df)"
R"lfs_eval$unc <- emulator$lfs$get_cov(eval_df)"
@rget lfs_eval
fs_eval = invlogit.(lfs_eval[:mean])
fs_eval_uc = invlogit.(lfs_eval[:mean] .+ 1.96 .* sqrt.(lfs_eval[:unc]))
fs_eval_lc = invlogit.(lfs_eval[:mean] .- 1.96 .* sqrt.(lfs_eval[:unc]))
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
obs_lc = logit(0.99*invlogit(obs))
obs_uc = logit(1.01*invlogit(obs))
target = Dict(:lfs => [obs_lc, obs_uc])
@rput target


R"new_points <- generate_new_runs(emulator, $n_test, target, method = 'lhs', cutoff = 3)"
@rget new_points;


l = @layout [a b]
pl1 = histogram(new_points[!,:b],legend=false,xlim=(lb[1],ub[1]),bins=lb[1]:0.00005:ub[1],title="NROY values for β")
vline!(pl1,[p[1]])
pl2 = histogram(new_points[!,:g],legend=false,xlim=(lb[2],ub[2]),bins=lb[2]:0.05:ub[2],title="NROY values for γ")
vline!(pl2,[p[2]])
plot(pl1, pl2, layout = l)

