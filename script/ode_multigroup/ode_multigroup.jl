
using ModelingToolkit
using OrdinaryDiffEq
using Plots;


K = 2;


@parameters t
D = Differential(t);


@parameters t β c[1:K] γ;


@variables (S(t))[1:K] (I(t))[1:K] (R(t))[1:K] (λ(t))[1:K]
state_eqs = [
       [D(S[i]) ~ -λ[i]*S[i],
        D(I[i]) ~ λ[i]*S[i]-γ*I[i],
        D(R[i]) ~ γ*I[i]]
            for i in 1:K]
# Turn into a 1D vector
state_eqs = vcat(state_eqs...);


@variables (N(t))[1:K] (p(t))[1:K,1:K]
variable_eqs = [
               [N[i] ~ S[i]+I[i]+R[i] for i in 1:K]...,
               [λ[i] ~ sum([β*c[i]*p[i,j]*I[j]/N[j] for j in 1:K]) for i in 1:K]...,
             [p[i,j] ~ c[j]*N[j]/sum([c[k]*N[k] for k in 1:K]) for j in 1:K for i in 1:K]...
               ];


@named sys = ODESystem([state_eqs;variable_eqs])


simpsys = structural_simplify(sys)


u₀ = [[S[i] => 990.0/K for i in 1:K]...,
      [I[i] => 10.0/K for i in 1:K]...,
      [R[i] => 0.0 for i in 1:K]...];


p = [β=>0.05, [c[i]=>10.0 for i in 1:K]..., γ=>0.25];


δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);


prob = ODEProblem(simpsys, u₀, tspan, p)
sol = solve(prob, Tsit5(), saveat=δt);


all_states = states(simpsys)


indexof(sym,syms) = findfirst(isequal(sym),syms)
S_indexes = [indexof(S[k],all_states) for k in 1:K]
I_indexes = [indexof(I[k],all_states) for k in 1:K]
R_indexes = [indexof(R[k],all_states) for k in 1:K];


Smat = sol[S_indexes,:]
Imat = sol[I_indexes,:]
Rmat = sol[R_indexes,:];


Stotal = sum(Smat,dims=1)'
Itotal = sum(Imat,dims=1)'
Rtotal = sum(Rmat,dims=1)';


times = sol.t
plot(times, Stotal, label="S", xlabel="Time", ylabel="Number")
plot!(times, Itotal, label="I")
plot!(times, Rtotal, label="R")


p2 = [β=>0.05, c[1] => 20, c[2] => 5, γ=>0.25]
prob2 = remake(prob, p=p2)
sol2 = solve(prob2, Tsit5(), saveat=δt);


plot(times, sol2(times, idxs=S_indexes)', labels=["S₁" "S₂"], linecolor=:blue, linestyle=[:solid :dash])
plot!(times, sol2(times, idxs=I_indexes)', labels=["I₁" "I₂"], linecolor=:red, linestyle=[:solid :dash])
plot!(times, sol2(times, idxs=R_indexes)', labels=["R₁" "R₂"], linecolor=:green, linestyle=[:solid :dash])
xlabel!("Time")
ylabel!("Number")

