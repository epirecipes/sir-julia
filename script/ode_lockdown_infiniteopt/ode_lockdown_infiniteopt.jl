
using InfiniteOpt
using Ipopt
using Plots;


β = 0.5 # infectivity rate
γ = 0.25 # recovery rate
υ_max = 0.5 # maximum intervention
υ_total = 10.0; # maximum cost


t0 = 0.0
tf = 100.0
dt = 0.1
extra_ts = collect(dt:dt:tf-dt);


S₀ = 0.99
I₀ = 0.01
C₀ = 0.00;


model = InfiniteModel(Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0);


@infinite_parameter(model, t ∈ [t0, tf], num_supports = length(extra_ts) + 2, 
                    derivative_method = OrthogonalCollocation(2))
add_supports(t, extra_ts);


@variable(model, S ≥ 0, Infinite(t))
@variable(model, I ≥ 0, Infinite(t))
@variable(model, C ≥ 0, Infinite(t));


@variable(model, 0 ≤ υ ≤ υ_max, Infinite(t), start = 0.0)
@constraint(model, υ_total_constr, ∫(υ, t) ≤ υ_total);


@objective(model, Min, C(tf));


@constraint(model, S(0) == S₀)
@constraint(model, I(0) == I₀)
@constraint(model, C(0) == C₀);


@constraint(model, S_constr, ∂(S, t) == -(1 - υ) * β * S * I)
@constraint(model, I_constr, ∂(I, t) == (1 - υ) * β * S * I - γ * I)
@constraint(model, C_constr, ∂(C, t) == (1 - υ) * β * S * I);


print(model)


optimize!(model)


termination_status(model)


S_opt = value(S, ndarray = true)
I_opt = value(I, ndarray = true)
C_opt = value(C, ndarray = true)
υ_opt = value(υ, ndarray = true)
obj_opt = objective_value(model)
ts = value(t);


t₁ = 14.338623046875002
t₂ = t₁ + υ_total/υ_max


plot(ts, S_opt, label="S", xlabel="Time", ylabel="Number")
plot!(ts, I_opt, label="I")
plot!(ts, C_opt, label="C")
plot!(ts, υ_opt, label="Optimized υ")
vspan!([t₁, t₂], color=:gray, alpha=0.5, label="Exact υ")

