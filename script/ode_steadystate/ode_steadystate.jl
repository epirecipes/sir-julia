
using ModelingToolkit
using OrdinaryDiffEq
using DifferentialEquations
using DiffEqCallbacks
using NonlinearSolve
using Random
using Distributions
using Plots
using LaTeXStrings
using DataFrames;


@parameters t β γ μ
@variables S(t) I(t)
D = Differential(t)
eqs = [D(S) ~ μ - β*S*I - μ*S,
       D(I) ~ β*S*I - (γ+μ)*I];


u₀ = [S => 0.99, I => 0.01]
p = [β => 0.5, γ => 0.25, μ => 0.025];


R₀ = β/(γ + μ)
substitute(R₀, p)


S₀ = 1/R₀
S₁ = substitute(S₀, p)
I₀ = (μ/β)*(R₀ - 1)
I₁ = substitute(I₀, p)
S₁, I₁


@named sys = ODESystem(eqs)
odeprob = ODEProblem(sys, u₀, (0, 50000), p)
odesol = solve(odeprob, RK4(); abstol = 1e-13, callback = TerminateSteadyState(1e-8, 1e-6));


times = odesol.t[1]:0.1:odesol.t[end]
odeout = Array(odesol(times))'
l = @layout [a b]
p1 = plot(times,
          odeout[:, 1],
          xlabel="Time",
          ylabel="Number",
          label="S")
plot!(p1,
      times,
      odeout[:, 2],
      label="I")
p2 = plot(odeout[:,1],
     odeout[:,2],
     xlabel=L"S",
     ylabel=L"I",
     legend=false,
     color=:black)
plot(p1, p2, layout=l)


ssprob = SteadyStateProblem(sys, u₀, p)
sssol = solve(ssprob, DynamicSS(RK4()); abstol=1e-13);


nlprob = NonlinearProblem(odeprob)
nlsol = solve(nlprob, NewtonRaphson())


Random.seed!(1234)
ninits = 4
results = [[nlprob.u0; nlsol]]
for i in 1:ninits
    newu₀ = rand(Dirichlet(3,1))[1:2]
    prob = remake(nlprob, u0=newu₀)
    sol = solve(prob, NewtonRaphson())
    push!(results, [newu₀; sol])
end
df = DataFrame(mapreduce(permutedims, vcat, results), :auto)
rename!(df, [:S₀, :I₀, :S₁, :I₁])
df

