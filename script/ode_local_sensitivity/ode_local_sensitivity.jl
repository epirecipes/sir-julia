
using OrdinaryDiffEq
using DiffEqSensitivity
using Zygote
using Plots


function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*c*I/N*S
        du[2] = β*c*I/N*S - γ*I
        du[3] = γ*I
    end
    nothing
end;


δt = 1.0
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax
num_timepoints = length(t);


u0 = [990.0,10.0,0.0] # S,I,R
num_states = length(u0);


p = [0.05,10.0,0.25]; # β,c,γ
num_params = length(p);


prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


sim_ode = (u0,p)-> solve(prob_ode,Tsit5(),u0=u0,p=p,saveat=t,sensealg=QuadratureAdjoint());


sol_ode = sim_ode(u0,p);


du0,dp = Zygote.jacobian(sim_ode,u0,p);


dβ = reshape(dp[:,1],(num_states,:))' # as β is the first parameter
dc = reshape(dp[:,2],(num_states,:))' # c is 2nd parameter
dγ = reshape(dp[:,3],(num_states,:))' # γ is 3rd parameter
dI₀ = reshape(du0[:,2],(num_states,:))'; # I₀ is the 2nd initial condition


plot(sol_ode.t,
     Array(sol_ode(t))',
     labels = ["S" "I" "R"],
     xlabel = "Time",
     ylabel = "Number")


l = @layout [a b; c d]
pl1 = plot(t,dβ,xlabel="Time",ylabel="dp",label=["S" "I" "R"],title="Sensitivity to β")
pl2 = plot(t,dc,xlabel="Time",ylabel="dp",label=["S" "I" "R"],title="Sensitivity to c")
pl3 = plot(t,dγ,xlabel="Time",ylabel="dp",label=["S" "I" "R"],title="Sensitivity to γ")
pl4 = plot(t,dI₀,xlabel="Time",ylabel="dp",label=["S" "I"  "R"],title="Sensitivity to I₀")
plot(pl1,pl2,pl3,pl4,layout=l)

