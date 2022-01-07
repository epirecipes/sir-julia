
using ApproxFun
using Plots
using BenchmarkTools


function sir_eqn(S,I,u0,p)
  (β,γ) = p
  (S0,I0) = u0
   return [S(0)-S0,
           I(0)-I0,
           S' + β*S*I,
           I' - β*S*I + γ*I]
end;


tmax = 40.0
t=Fun(identity, 0..tmax);


u0 = [990.0,10.0]; # S,I


p = [0.0005,0.25]; # β,γ


S,I = newton((S,I)->sir_eqn(S,I,u0,p), u0 .* one(t); maxiterations=50);


plot(S,label="S",xlabel="Time",ylabel="Number")
plot!(I,label="I")


@benchmark newton((S,I)->sir_eqn(S,I,u0,p), u0 .* one(t); maxiterations=50)

