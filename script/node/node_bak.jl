
using OrdinaryDiffEq
using DiffEqFlux, Flux
using Random
using Plots;


Random.seed!(123);


function sir_ode(u,p,t)
    (s,i,r) = u
    (β,γ) = p
    ds = -β*s*i
    di = β*s*i - γ*i
    dr = γ*i
    [ds,di,dr]
end;


p = [0.5,0.25]
u0 = [0.99, 0.01, 0.0]
tspan = (0.0, 40.0)
δt = 1;


solver = Rodas5();


sir_prob = ODEProblem(sir_ode, u0, tspan, p)
sir_sol = solve(sir_prob, solver, saveat = δt);


train_time = 30.0
train_data = Array(sir_sol(0:δt:train_time));


nhidden = 8
sir_node = FastChain(FastDense(3, nhidden, tanh),
                     FastDense(nhidden, nhidden, tanh),
                     FastDense(nhidden, nhidden, tanh),
                     FastDense(nhidden, 3));


p_ = Float64.(initial_params(sir_node));
function dudt_sir_node(u,p,t)
    s,i,r = u
    ds,di,dr = ann_node([s,i,r],p)
    [ds,di,dr]
end
prob_node = ODEProblem(dudt_sir_node, u0, tspan, p_);


tspan_train = (0,train_time)
prob_node = NeuralODE(sir_node,
                      tspan_train,
                      solver,
                      saveat=δt,
                      sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))
nump = length(prob_node.p)


function loss(p)
    sol = prob_node(u0,p)
    pred = Array(sol)
    sum(abs2, (train_data .- pred)), pred
end;


const losses = []
callback = function (p, l, pred)
    push!(losses, l)
    numloss = length(losses)
    if numloss % 50 == 0
        display("Epoch: " * string(numloss) * " Loss: " * string(l))
    end
    return false
end;


res_node = DiffEqFlux.sciml_train(loss,
                                   prob_node.p,
                                   cb = callback);


plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss", legend = false)


prob_node = NeuralODE(sir_node,
                      tspan_train,
                      solver,
                      saveat=δt,
                      sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
                      p = res_node)
sol_node = prob_node(u0);


scatter(sir_sol, label=["True Susceptible" "True Infected" "True Recovered"])
plot!(sol_node, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])


tspan_test = (0.0, 40.0)
prob_node_test = NeuralODE(sir_node,
                      tspan_test,
                      solver,
                      saveat=δt,
                      sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
                      p = res_node)
sol_node_test = prob_node_test(u0);


p_node = scatter(sol_node_test, legend = :topright, label=["True Susceptible" "True Infected" "True Recovered"], title="Neural ODE Extrapolation: training until t=30")
plot!(p_node,sol_node_test, lw=5, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])


newu0 = [0.95, 0.05, 0.0]
sir_prob_u0 = remake(sir_prob,u0=newu0)
sir_sol_u0 = solve(sir_prob_u0, solver, saveat =  δt)
node_sol_u0 = prob_node(newu0)
p_node = scatter(sir_sol_u0, legend = :topright, label=["True Susceptible" "True Infected" "True Recovered"], title="Neural ODE with different initial conditions")
plot!(p_node,node_sol_u0, lw=5, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])


tspan_train2 = (0.0,20.0)
prob2 = ODEProblem(sir_ode, u0, tspan_train2, p)
sol2 = solve(prob2, solver, saveat = δt)
data2 = Array(sol2)
solver2 = ExplicitRK()
prob_node2 = NeuralODE(sir_node,
                      tspan_train2,
                      solver2,
                      saveat=δt,
                      sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))
function loss2(p)
    sol = prob_node2(u0,p)
    pred = Array(sol)
    sum(abs2, (data2 .- sol)), pred
end
const losses2 = []
callback2 = function (p, l, pred)
    push!(losses2, l)
    numloss = length(losses2)
    if numloss % 50 == 0
        display("Epoch: " * string(numloss) * " Loss: " * string(l))
    end
    return false
end
res_node2 = DiffEqFlux.sciml_train(loss2,
                                   prob_node2.p,
                                   cb = callback2);


prob_node2_test = NeuralODE(sir_node,
                      tspan_test,
                      solver,
                      saveat=δt,
                      sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
                      p = res_node2)
sol_node2_test = prob_node2_test(u0);


p_node2 = scatter(sir_sol, legend = :topright, label=["True Susceptible" "True Infected" "True Recovered"], title="Neural ODE Extrapolation: training until t=20")
plot!(p_node2, sol_node2_test, lw=5, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])


function loss3(p)
    sol = prob_node(u0,p)
    pred = Array(sol)
    sum(abs2, (train_data[2,:] .- sol[2,:])), pred
end
const losses3 = []
callback3 = function (p, l, pred)
    push!(losses3, l)
    numloss = length(losses3)
    if numloss % 50 ==0
        display("Epoch: " * string(numloss) * " Loss: " * string(l))
    end
    return false
end
res_node3 = DiffEqFlux.sciml_train(loss3,
                                   prob_node.p,
                                   cb = callback3);


prob_node3_test = NeuralODE(sir_node,
                      tspan_test,
                      solver,
                      saveat=δt,
                      sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
                      p = res_node3)
sol_node3_test = prob_node3_test(u0);


p_node3 = scatter(sir_sol, legend = :topright, label=["True Susceptible" "True Infected" "True Recovered"], title="Neural ODE Extrapolation: training on I(t) until t=30")
plot!(p_node3, sol_node3_test, lw=5, label=["Estimated Susceptible" "Estimated Infected" "Estimated Recovered"])

