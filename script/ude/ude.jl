
using OrdinaryDiffEq
using Distributions
using DiffEqFlux, Flux
using Random
using Plots;


Random.seed!(123);


function sir_ode(u,p,t)
    (S,I,C) = u
    (β,γ) = p
    dS = -β*S*I
    dI = β*S*I - γ*I
    dC = β*S*I
    [dS,dI,dC]
end;


solver = RadauIIA3();


N = 1000.0
p = [0.5,0.25]
u0 = [0.99, 0.01, 0.0]
tspan = (0., 40.)
δt = 1;


sir_prob = ODEProblem(sir_ode, u0, tspan, p)
sir_sol = solve(sir_prob, solver, saveat = δt);


plot(sir_sol,
     xlabel = "Time",
     ylabel = "Proportion",
     labels = ["S" "I" "R"])


train_time = 30.0
tsdata = Array(sir_sol(0:δt:train_time))
cdata = diff(tsdata[3,:])
noisy_data = rand.(Poisson.(N .* cdata));


plot(1:δt:train_time, N .* cdata,
     xlabel = "Time",
     ylabel = "New cases per day",
     label = "True value")
scatter!(1:δt:train_time, noisy_data, label="Data")


foi1 = FastDense(1, 1, relu, bias=false)
p1_ = Float64.(initial_params(foi1))
length(p1_)


function sir_ude(u,p_,t,foi)
    S,I,C = u
    β,γ = p
    λ = foi([I],p_)[1]
    dS = -λ*S
    dI = λ*S - γ*I
    dC = λ*S
    [dS, dI, dC]
end;


tspan_train = (0,train_time)
sir_ude1 = (u,p_,t) -> sir_ude(u,p_,t,foi1)
prob_ude1 = ODEProblem(sir_ude1,
                      u0,
                      tspan_train,
                      p1_);


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


loss(prob_ude1.p, prob_ude1);


const losses1 = []
callback1 = function (p, l, pred)
    push!(losses1, l)
    numloss = length(losses1)
    if numloss % 10 == 0
        display("Epoch: " * string(numloss) * " Loss: " * string(l))
    end
    return false
end;


res_ude1 = DiffEqFlux.sciml_train((θ)->loss(θ,prob_ude1),
                                  p1_,
                                  cb=callback1);


res_ude1.minimizer, losses1[end]


plot(losses1, xaxis = :log, xlabel = "Iterations", ylabel = "Loss", legend=false)


prob_ude1_fit = ODEProblem(sir_ude1, u0, tspan, res_ude1.minimizer)
sol_ude1_fit = solve(prob_ude1_fit, solver, saveat = δt)
scatter(sir_sol, label=["True Susceptible" "True Infected" "True Recovered"],title="Fitted true model")
plot!(sol_ude1_fit, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])


Imax = maximum(tsdata[2,:])
Igrid = 0:0.01:0.5
λ = [foi1([I],res_ude1.minimizer)[1] for I in Igrid]
scatter(Igrid,λ,xlabel="Proportion of population infected, I",ylab="Force of infection, λ",label="Neural network prediction")
Plots.abline!(p[1],0,label="True value")
Plots.vline!([Imax],label="Upper bound of training data")


Random.seed!(1234)
nhidden = 4
foi2 = FastChain(FastDense(1, nhidden, relu),
                     FastDense(nhidden, nhidden, relu),
                     FastDense(nhidden, 1, relu))
p2_ = Float64.(initial_params(foi2))
length(p2_)


sir_ude2 = (u,p_,t) -> sir_ude(u,p_,t,foi2)
prob_ude2 = ODEProblem(sir_ude2,
                      u0,
                      tspan_train,
                      p2_);


const losses2 = []
callback2 = function (p, l, pred)
    push!(losses2, l)
    numloss = length(losses2)
    if numloss % 10 == 0
        display("Epoch: " * string(numloss) * " Loss: " * string(l))
    end
    return false
end;


res_ude2 = DiffEqFlux.sciml_train((θ)->loss(θ,prob_ude2),
                                  p2_,
                                  cb = callback2);


losses1[end],losses2[end]


prob_ude2_fit = ODEProblem(sir_ude2, u0, tspan, res_ude2.minimizer)
sol_ude2_fit = solve(prob_ude2_fit, solver, saveat = δt)
scatter(sir_sol, label=["True Susceptible" "True Infected" "True Recovered"],title="Fitted UDE model")
plot!(sol_ude2_fit, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])


λ = [foi2([I],res_ude2.minimizer)[1] for I in Igrid]
scatter(Igrid, λ, xlabel="Proportion of population infected, i", ylab="Force of infection, λ", label="Neural network prediction")
Plots.abline!(p[1], 0,label="True value")
Plots.vline!([Imax], label="Upper bound of training data")

