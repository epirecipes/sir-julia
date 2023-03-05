
using OrdinaryDiffEq
using DiffEqCallbacks
using Integrals
using Optimization
using OptimizationOptimJL
using Plots;


function sir_ode!(du,u,p,t)
    (S, I, C) = u
    (β, γ, υ) = p
    @inbounds begin
        du[1] = -β*(1-υ)*S*I
        du[2] = β*(1-υ)*S*I - γ*I
        du[3] = β*(1-υ)*S*I
    end
    nothing
end;


function simulate(p, u0, t₁, dur, ss, alg)
    t₂ = t₁ + dur
    lockdown_times = [t₁, t₂]
    β, γ, υ = p
    function affect!(integrator)
        if integrator.t < lockdown_times[2]
            integrator.p[3] = υ
        else
            integrator.p[3] = 0.0
        end
    end
    cb = PresetTimeCallback(lockdown_times, affect!)
    tspan = (0.0, t₂+ss)
    # Start with υ=0   
    prob = ODEProblem(sir_ode!, u0, tspan, [β, γ, 0.0])
    sol = solve(prob, alg, callback = cb)
    return sol
end;


function final_size(p, u0, t₁, dur, ss, alg)
    sol = simulate(p, u0, t₁, dur, ss, alg)
    return sol[end][3]
end;


u0 = [0.99, 0.01, 0.0];


dur = 20.0
p = [0.5, 0.25, 0.5]; # β, γ, υ


ss = 100.0
ts = collect(0.0:0.1:100.0);


alg = Tsit5();


p1 = copy(p)
p1[3] = 0.0
tf = 1000
prob1 = ODEProblem(sir_ode!, u0, (0.0, tf), p1)
sol1 = solve(prob1, alg);


sol1[end][3]


plot(sol1,
     xlim=(0, ss),
     labels=["S" "I" "C"],
     xlabel="Time",
     ylabel="Number")


pk(u,p) = - sol1(u[1])[2]
pkprob = OptimizationProblem(pk, [20.0])
pksol = solve(pkprob, NelderMead());


t₁ = pksol[1]
sol2 = simulate(p, u0, t₁, dur, ss, alg);


sol2[end][3]


plot(sol2, xlim=(0, 100.0))


fs(u, p_) = final_size(p, u0, u[1], dur, ss, alg);


fsprob = OptimizationProblem(fs, [t₁])
fssol = solve(fsprob, NelderMead())
t₁ = fssol[1]


fs(fssol,[])


final_sizes = [fs([x], []) for x in ts]
plot(ts,
     final_sizes,
     xlabel="Time of intervention, t₁",
     ylabel="Final size",
     ylim=(0,1),
     xlim=(0,40),
     legend=false)
vline!(fssol)


υ = zeros(length(ts))
t₂ = t₁ + dur
[υ[i]=p[3] for i in 1:length(ts) if (ts[i] > t₁ && ts[i] <= t₂)]; # t ⋵ (t₁, t₂]


t₁ = fssol[1]
sol = simulate(p, u0, t₁, dur, ss, alg)
plot(sol,
     xlim=(0, ss),
     labels=["S" "I" "C"],
     xlabel="Time",
     ylabel="Number")
plot!(ts, υ, label="υ")

