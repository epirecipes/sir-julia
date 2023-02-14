
using ModelingToolkit
using OrdinaryDiffEq
using Plots


K = 4
@parameters t β c δ
@variables S(t) (I(t))[1:K] R(t)
D = Differential(t)
ΣI = sum(I[1:K])
N=S+ΣI+R
eqs = [D(S) ~ -β*c*ΣI/N*S,
       D(I[1]) ~ β*c*ΣI/N*S-δ*I[1],
       [D(I[i]) ~ δ*I[i-1] - δ*I[i] for i in 2:K]...,
       D(R) ~ δ*I[K]];
@named sys = ODESystem(eqs);


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);


u0 = [S => 990.0,
      I[1] => 10.0,
      [I[i] => 0.0 for i in 2:K]...,
      R => 0.0];


p = [β=>0.05,
    c=>10.0,
    δ=>0.25*K];


prob_ode = ODEProblem(sys,u0,tspan,p;jac=true)
sol_ode = solve(prob_ode, Tsit5(), saveat=δt);


out = Array(sol_ode)
Isum = vec(sum(out[2:(K+1),:],dims=1));


plot(sol_ode.t, out[1,:], xlabel="Time", ylabel="Number", label="S")
plot!(sol_ode.t, Isum, label="I")
plot!(sol_ode.t, out[end,:], label="R")

