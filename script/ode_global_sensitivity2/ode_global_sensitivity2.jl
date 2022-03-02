
using OrdinaryDiffEq
using DiffEqCallbacks
using GlobalSensitivity
using Distributions
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
tspan = (0.0,tmax);


cb_ss = TerminateSteadyState();


N = 1000.0;
u0 = [990.0,10.0,0.0];
p = [0.05,10.0,0.25]; # β,c,γ


n_samples = 1000 # Number of samples
# Parameters are β, c, γ, I₀
lb = [0.01, 5.0, 0.1, 1.0]
ub = [0.1, 20.0, 1.0, 50.0]
n_params = 4;


prob_ode = ODEProblem(sir_ode!,u0,tspan,p);


f1 = function(pu0)
  p = pu0[1:3]
  I0 = pu0[4]
  u0 = [N-I0,I0,0.0]
  prob = remake(prob_ode;p=p,u=u0)
  sol = solve(prob, ROS34PW3(),callback=cb_ss)
  [maximum(sol[2,:]), sol.t[argmax(sol[2,:])], sol[end][3]]
end;


m_morris = gsa(f1, Morris(num_trajectory=n_samples), [[lb[i],ub[i]] for i in 1:n_params]);


m_morris.means


m_morris.variances


m_sobol = gsa(f1, Sobol(), [[lb[i],ub[i]] for i in 1:n_params],N=n_samples);


m_sobol.ST


m_sobol.S1


m_regression = gsa(f1, RegressionGSA(rank=true), [[lb[i],ub[i]] for i in 1:n_params]; samples = n_samples);


m_regression.partial_correlation


m_regression.partial_rank_correlation


m_efast = gsa(f1, eFAST(), [[lb[i],ub[i]] for i in 1:n_params]; n = n_samples);


m_efast.ST


m_efast.S1


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


m_efast_parallel = gsa(pf1, eFAST(), [[lb[i],ub[i]] for i in 1:n_params]; n = n_samples, batch = true);

