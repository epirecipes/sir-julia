
using OrdinaryDiffEq
using DiffEqCallbacks
using DataInterpolations
using Distributions
using DiffEqFlux, Flux
using Random
using Plots;


Random.seed!(123);


function sira_ode(u,p,t)
    (S,I,C) = u
    (β,γ,α) = p
    dS = -β*S*(I^α)
    dI = β*S*(I^α) - γ*I
    dC = β*S*(I^α)
    [dS,dI,dC]
end;


solver = ROS34PW3();


N = 1000.0
p = [0.5, 0.25, 0.9]
u0 = [0.99, 0.01, 0.0]
tspan = (0., 40.)
δt = 1;


sira_prob = ODEProblem(sira_ode, u0, tspan, p)
sira_sol = solve(sira_prob, solver, saveat = δt);


train_time = 30.0
tsdata = Array(sira_sol(0:δt:train_time))
cdata = diff(tsdata[3,:])
noisy_data = rand.(Poisson.(N .* cdata));


tt = 0:δt:train_time
plot(tt[2:end],
     N .* cdata,
     xlabel = "Time",
     ylabel = "Number of new infected",
     label = "Model")
scatter!(tt,
         noisy_data,
         label = "Simulated data")


function sir_ude(u,p_,t,foi)
    S,I,C = u
    β,γ,α = p
    λ = foi([I],p_)[1]
    dS = -λ*S
    dI = λ*S - γ*I
    dC = λ*S
    [dS, dI, dC]
end;


function foi(ivec,p)
    t = 0:0.1:1
    f = LinearInterpolation([0.0;exp.(p)],t)
    return [f(ivec[1])]
end
p_ = log.(0.6 .* collect(0.1:0.1:1));


sir_psm = (u,p_,t) -> sir_ude(u,p_,t,foi)
prob_psm = ODEProblem(sir_psm,
                      u0,
                      (0.0, train_time),
                      p_);


function predict(θ, prob)
    Array(solve(prob,
                solver;
                u0 = u0,
                p = θ,
                saveat = δt,
                sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end;


function loss(θ, prob)
    pred = predict(θ, prob)
    cpred = abs.(N*diff(pred[3,:]))
    Flux.poisson_loss(cpred, float.(noisy_data)), cpred
end;


const losses = []
callback = function (p, l, pred)
    push!(losses, l)
    numloss = length(losses)
    if numloss % 20 == 0
        display("Epoch: " * string(numloss) * " Loss: " * string(l))
    end
    return false
end;


res_psm = DiffEqFlux.sciml_train((θ)->loss(θ,prob_psm),
                                  p_,
                                  cb = callback);


prob_psm_fit = ODEProblem(sir_psm, u0, tspan, res_psm.minimizer)
sol_psm_fit = solve(prob_psm_fit, solver, saveat = δt)
scatter(sira_sol, label=["True Susceptible" "True Infected" "True Recovered"],title="Fitted partially specified model")
plot!(sol_psm_fit, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])
Plots.vline!([train_time],label="Training time")


Imax = maximum(tsdata[2,:])
Igrid = 0:0.01:1.0 # create a fine grid
β,γ,α = p
λ_true = β .* Igrid.^α
λ = [foi([I], res_psm.minimizer)[1] for I in Igrid]
scatter(Igrid,
        λ,
        xlabel="Proportion of population infected, I",
        ylab="Force of infection, λ",
        label="Model prediction")
Plots.vline!([Imax], color=:orange, label="Upper bound of training data")
plot!(Igrid, λ_true, color=:red, label="True function")

