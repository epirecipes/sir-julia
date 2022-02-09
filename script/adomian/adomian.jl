
using DifferentialEquations
using SimpleDiffEq
using Polynomials
using ModelingToolkit
using Symbolics
using Plots


M = 2
N = 10
f(s,i) = s*i;


@variables a[1:M,1:N];


An = [] # Empty array of Adomian
A₀ = f(a[1,1],a[2,1]) # s₀*i₀
push!(An,A₀)
for n = 1:(N-1)
  A = 0
  for i = 1:M
    for k = 1:n
      A += k*a[i,k+1]*Symbolics.derivative(An[end],a[i,k])
    end
  end
  A /= n
  push!(An,A)
end;


tspan = (0.0,8.0)
trange = 0:0.1:8;


u0 = [20.0 15.0]'; # S, I


p = [0.01, 0.02]; # β, γ


(S,I) = u0'
v = u0
β, γ = p'
Ii = integrate(Polynomial([I],:t))
F = eval(build_function(An[1],a))
SIi = integrate(Polynomial([F(v)],:t))
for i in 1:(N-1)
  vv = [Polynomial(ones(N),:t) Polynomial(ones(N),:t)]'
  vv[1,1] = -β*SIi
  vv[2,1] =  β*SIi - γ*Ii
  global v = hcat(v,vv)
  # Now update
  global I = v[2,i+1]
  global Ii = integrate(I)
  global F = eval(build_function(An[i+1],a))
  global SIi = integrate(F(v))
end


S = sum(v[1,1:end])


I = sum(v[2,1:end])


Sa = [S(t) for t in trange]
Ia = [I(t) for t in trange];


plot(trange,Sa,label="S")
plot!(trange,Ia,label="I")


function sir_ode!(du,u,p,t)
    (S,I) = u
    (β,γ) = p
    @inbounds begin
        du[1] = -β*S*I
        du[2] = β*S*I - γ*I
    end
    nothing
end
prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode)
plot(sol_ode,
     label=["S" "I"],
     xlabel="Time",
     ylabel="Number")


Fv = [eval(build_function(An[i],a)) for i in 1:N];


function SIR(u0,p,trange,Fv)
  (S,I) = u0'
  v = u0
  β, γ = p'
  Ii = integrate(Polynomial([I],:t))
  SIi = integrate(Polynomial([Fv[1](v)],:t))
  for i in 1:(N-1)
    vv = [Polynomial(ones(N),:t) Polynomial(ones(N),:t)]'
    vv[1,1] = -β*SIi
    vv[2,1] =  β*SIi - γ*Ii
    v = hcat(v,vv)
    # Now update
    I = v[2,i+1]
    Ii = integrate(I)
    SIi = integrate(Fv[i+1](v))
  end
  S = sum(v[1,1:end])
  I = sum(v[2,1:end])
  Sa = [S(t) for t in trange]
  Ia = [I(t) for t in trange]
  return [trange Sa Ia]
end;


u0 = [990.0 10.0]'
p = [0.0005 0.25]
trange = 0:0.1:40
sol1 = SIR(u0,p,trange,Fv)
plot(sol1[1:end,1],sol1[1:end,2:3],
     label=["S" "I"],
     xlabel="Time",
     ylabel="Number")


u = u0
t = 0.0
sol = [[0.0 u']]
for i in 1:40
  s = SIR(u,p,0.0:0.1:1.0,Fv)
  s[1:end,1] .+= t
  global t += 1.0
  push!(sol,s[2:end,:])
  global u = s[end,2:3]
end
sol = vcat(sol...);


plot(sol[1:end,1],sol[1:end,2:3],
     label=["S" "I"],
     xlabel="Time",
     ylabel="Number")


prob_ode2 = ODEProblem(sir_ode!,u0,(0,40.0),p)
sol_ode2 = solve(prob_ode2)
plot(sol_ode2,
     label=["S" "I"],
     xlabel="Time",
     ylabel="Number")

