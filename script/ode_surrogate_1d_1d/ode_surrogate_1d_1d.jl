
using OrdinaryDiffEq
using DiffEqCallbacks
using Surrogates
using Random
using Optim
using Plots
using BenchmarkTools


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
tspan = (0.0,tmax);


cb_ss = TerminateSteadyState();


n_train = 25 # Number of samples
n_test = 1000; # Number of samples


# Parameters are β
lb = 0.00005
ub = 0.001;


N = 1000.0
u0 = [990.0,10.0,0.0]
p = [0.0005,0.25]
prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


Random.seed!(123)
sampler = LatinHypercubeSample();


β = Surrogates.sample(n_train,lb,ub,sampler)
β₀ = copy(β);


final_size = function(β)
  p = prob_ode.p
  p[1] = β
  prob = remake(prob_ode;p=p)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  fsp = sol[end][3]/N
  fsp
end;


fs = final_size.(β);


β_grid = lb:0.00001:ub
plot(β_grid,final_size.(β_grid),label="True function",xlab="Infectivity parameter, β",ylab="Final size")
scatter!(β,fs,legend=:right,label="Sampled points")


fs_surrogate = RadialBasis(β, fs, lb, ub, rad=cubicRadial);


scatter(fs,
        fs_surrogate.(β),
        legend=false,
        xlabel="True final size",
        ylabel="Surrogate final size")
Plots.abline!(1.0,0.0)


surrogate_optimize(final_size,
                   SRBF(),
                   lb,
                   ub,
                   fs_surrogate,
                   sampler;
                   maxiters=1000,
                   num_new_samples=1000);


β₁ = copy(β)
length(β₀),length(β)


fs_opt = final_size.(β)
scatter(fs_opt,fs_surrogate.(β), xlabel="Final size", ylabel="Surrogate final size",legend=false)
Plots.abline!(1.0, 0.0)


β_test = sample(n_test,lb,ub,UniformSample())
fs_test = final_size.(β_test)
fs_test_pred = fs_surrogate.(β_test);


l = @layout [a b]
pl1 = plot(β_grid,final_size.(β_grid),color=:red,label="Model",xlabel="β",ylabel="Final size",legend=:right)
scatter!(pl1, β_test,fs_test_pred,color=:blue,label="Surrogate")
pl2 = scatter(fs_test,fs_test_pred,color=:blue,legend=false,xlabel="Final size",ylabel="Surrogate final size")
Plots.abline!(pl2,1.0,0.0)
plot(pl1,pl2,layout=l)


@benchmark final_size(0.0005)


@benchmark fs_surrogate(0.0005)


peak_infected = function(β)
  p = prob_ode.p
  p[1] = β
  prob = remake(prob_ode;p=p)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  tss = sol.t[end]
  f = (t) -> -sol(t,idxs=2)
  opt = Optim.optimize(f,0.0,tss)
  pk = -opt.minimum
  pk
end


plot(β_grid, peak_infected,xlabel="β",ylabel="Peak infected",legend=false)


pk = peak_infected.(β)
pk_surrogate = RadialBasis(β,pk,lb,ub,rad = cubicRadial)
surrogate_optimize(peak_infected,
                   SRBF(),
                   lb,
                   ub,
                   pk_surrogate,
                   sampler;
                   maxiters=1000,
                   num_new_samples=1000)
pk_test = peak_infected.(β_test)
pk_test_pred = pk_surrogate.(β_test);


l = @layout [a b]
pl1 = plot(β_grid,peak_infected.(β_grid),color=:red,label="Model",xlabel="β",ylabel="Peak infected",legend=:right)
scatter!(pl1, β_test,pk_test_pred,color=:blue,label="Surrogate")
pl2 = scatter(pk_test,pk_test_pred,color=:blue,legend=false,xlabel="Peak infected",ylabel="Surrogate peak infected")
Plots.abline!(pl2,1.0,0.0)
plot(pl1,pl2,layout=l)


@benchmark peak_infected(0.0005)


@benchmark pk_surrogate(0.0005)


peak_time = function(β)
  p = prob_ode.p
  p[1] = β
  prob = remake(prob_ode;p=p)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  tss = sol.t[end]
  f = (t) -> -sol(t,idxs=2)
  opt = Optim.optimize(f,0.0,tss)
  pt = opt.minimizer
  pt
end


plot(β_grid,peak_time,xlabel="β",ylabel="Peak time",legend=false)


pt = peak_time.(β)
pt_surrogate = RadialBasis(β,pt,lb,ub,rad = cubicRadial)
surrogate_optimize(peak_time,
                   SRBF(),
                   lb,
                   ub,
                   pt_surrogate,
                   sampler;
                   maxiters=1000,
                   num_new_samples=1000)
pt_test = peak_time.(β_test)
pt_test_pred = pt_surrogate.(β_test);


l = @layout [a b]
pl1 = plot(β_grid,peak_time.(β_grid),color=:red,label="Model",xlabel="β",ylabel="Peak time",legend=:right)
scatter!(pl1, β_test,pt_test_pred,color=:blue,label="Surrogate")
pl2 = scatter(pt_test,pt_test_pred,color=:blue,legend=false,xlabel="Peak time",ylabel="Surrogate peak time")
Plots.abline!(pl2,1.0,0.0)
plot(pl1,pl2,layout=l)


@benchmark peak_time(0.0005)


@benchmark pt_surrogate(0.0005)

