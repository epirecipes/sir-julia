
using OrdinaryDiffEq
using DiffEqCallbacks
using QuasiMonteCarlo
using StatsBase
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


tmax = 10000.0
tspan = (0.0,tmax)


cb_ss = TerminateSteadyState();


N = 1000.0;
u0 = [990.0,10.0,0.0];
p = [0.05,10.0,0.25]; # β,c,γ


n_samples = 1000 # Number of samples
# Parameters are β, c, γ, I₀
lb = [0.01, 5.0, 0.1, 1.0]
ub = [0.1, 20.0, 1.0, 50.0];


pu0 = QuasiMonteCarlo.sample(n_samples,lb,ub,LatinHypercubeSample());


prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


f1 = function(pu0)
  p = pu0[1:3]
  I0 = pu0[4]
  u0 = [N-I0,I0,0.0]
  prob = remake(prob_ode;p=p,u=u0)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  [maximum(sol[2,:]), sol.t[argmax(sol[2,:])], sol[end][3]]
end;


m_serial = [f1(pu0[:,i]) for i in 1:n_samples]
m_serial = hcat(m_serial...); # convert into matrix


pf1 = function (pu0)
  p = pu0[1:3,:]
  I0 = pu0[4,:]
  prob_func(prob,i,repeat) = remake(prob;p=p[:,i],u=[N-I0[i],I0[i],0.0])
  ensemble_prob = EnsembleProblem(prob_ode,prob_func=prob_func)
  sol = solve(ensemble_prob,ROS34PW3(),EnsembleThreads();trajectories=size(p,2))
  out = zeros(3,size(p,2))
  for i in 1:size(p,2)
    out[1,i] = maximum(sol[i][2,:])
    out[2,i] = sol[i].t[argmax(sol[i][2,:])]
    out[3,i] = sol[i][end][3]
  end
  out
end;


m_parallel = pf1(pu0);


l = @layout [a b; c d]
pl1 = scatter(pu0[1,:],m_parallel[1,:],title="Peak infected",xlabel="β",ylabel="Number")
pl2 = scatter(pu0[2,:],m_parallel[1,:],title="Peak infected",xlabel="c",ylabel="Number")
pl3 = scatter(pu0[3,:],m_parallel[1,:],title="Peak infected",xlabel="γ",ylabel="Number")
pl4 = scatter(pu0[4,:],m_parallel[1,:],title="Peak infected",xlabel="I₀",ylabel="Number")
plot(pl1,pl2,pl3,pl4,layout=l,legend=false)


l = @layout [a b; c d]
pl1 = scatter(pu0[1,:],m_parallel[2,:],title="Peak time",xlabel="β",ylabel="Time")
pl2 = scatter(pu0[2,:],m_parallel[2,:],title="Peak time",xlabel="c",ylabel="Time")
pl3 = scatter(pu0[3,:],m_parallel[2,:],title="Peak time",xlabel="γ",ylabel="Time")
pl4 = scatter(pu0[4,:],m_parallel[2,:],title="Peak time",xlabel="I₀",ylabel="Time")
plot(pl1,pl2,pl3,pl4,layout=l,legend=false)


l = @layout [a b; c d]
pl1 = scatter(pu0[1,:],m_parallel[3,:],title="Final size",xlabel="β",ylabel="Number")
pl2 = scatter(pu0[2,:],m_parallel[3,:],title="Final size",xlabel="c",ylabel="Number")
pl3 = scatter(pu0[3,:],m_parallel[3,:],title="Final size",xlabel="γ",ylabel="Number")
pl4 = scatter(pu0[4,:],m_parallel[3,:],title="Final size",xlabel="I₀",ylabel="Number")
plot(pl1,pl2,pl3,pl4,layout=l,legend=false)


corkendall(pu0',m_parallel')

