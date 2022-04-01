
using OrdinaryDiffEq
using DataDrivenDiffEq
using ModelingToolkit
using Distributions
using Random
using Plots


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


solver = ExplicitRK();


sir_prob = ODEProblem(sir_ode, u0, tspan, p)
sir_sol = solve(sir_prob, solver, saveat = δt);


dd_prob = ContinuousDataDrivenProblem(sir_sol);


@parameters t
@variables u[1:3](t)
Ψ = Basis([u; u[1]*u[2]], u, independent_variable = t)


res_koopman = solve(dd_prob, Ψ, DMDPINV())
sys_koopman = result(res_koopman);


equations(sys_koopman)


parameter_map(res_koopman)


res_sindy = solve(dd_prob, Ψ, STLSQ(),digits=1)
sys_sindy = result(res_sindy);


equations(sys_sindy)


parameter_map(res_sindy)


sir_data = Array(sir_sol);


A = 500.0 # Smaller values of A = noisier data
noisy_data = sir_data
# Note that we can't draw from u0 as R(0)=0
for i in 2:size(sir_data)[2]
    noisy_data[1:3,i] = rand(Dirichlet(A*sir_data[1:3,i]))
end;


scatter(sir_sol.t,noisy_data',title="Noisy data",xlabel="Time",ylabel="Proportion",labels=["S+noise" "I+noise" "R+noise"])
plot!(sir_sol,labels=["S" "I" "R"],legend=:left)


noisy_dd_prob = ContinuousDataDrivenProblem(noisy_data,sir_sol.t,GaussianKernel());


noisy_res_koopman = solve(noisy_dd_prob, Ψ, DMDPINV())
noisy_sys_koopman = result(noisy_res_koopman)
equations(noisy_sys_koopman), parameter_map(noisy_res_koopman)


noisy_res_sindy = solve(noisy_dd_prob, Ψ, STLSQ())
noisy_sys_sindy = result(noisy_res_sindy)
equations(noisy_sys_sindy), parameter_map(noisy_res_sindy)


A = 50.0 # Smaller values of A = noisier data
v_noisy_data = sir_data
# Note that we can't draw from u0 as R(0)=0
for i in 2:size(sir_data)[2]
    v_noisy_data[1:3,i] = rand(Dirichlet(A*sir_data[1:3,i]))
end;


scatter(sir_sol.t,v_noisy_data',title="Very noisy data",xlabel="Time",ylabel="Proportion",labels=["S+noise" "I+noise" "R+noise"])
plot!(sir_sol,labels=["S" "I" "R"],legend=:left)


v_noisy_dd_prob = ContinuousDataDrivenProblem(v_noisy_data,sir_sol.t,GaussianKernel())
v_noisy_res_sindy = solve(v_noisy_dd_prob, Ψ, STLSQ())
v_noisy_sys_sindy = result(v_noisy_res_sindy)
equations(v_noisy_sys_sindy), parameter_map(v_noisy_res_sindy)

