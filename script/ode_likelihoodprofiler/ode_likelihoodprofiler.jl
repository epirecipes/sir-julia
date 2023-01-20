
using OrdinaryDiffEq
using LikelihoodProfiler
using Random
using Distributions
using Optim
using QuasiMonteCarlo # for Latin hypercube sampling
using Plots # for plotting output
using DataFrames # for formatting results


function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end;


δt = 1.0
tmax = 40.0
tspan = (0.0, tmax);


u0 = [990.0, 10.0, 0.0, 0.0]; # S, I, R, C


p = [0.05, 10.0, 0.25]; # β, c, γ


prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), saveat=δt);


sol_plot = plot(sol_ode,
                plotdensity=1000,
                xlabel = "Time",
                ylabel = "Number",
                labels = ["S" "I" "R" "C"])


out = Array(sol_ode)
C = out[4,:]
X = C[2:end] .- C[1:(end-1)];


Random.seed!(1234)
data = rand.(Poisson.(X));


function minustwologlik(θ)
    (i₀, β) = θ
    I = i₀*1000.0
    prob = remake(prob_ode, u0=[1000.0-I, I, 0.0, 0.0], p=[β, 10.0, 0.25])
    sol = solve(prob, Tsit5(), saveat=δt)
    out = Array(sol)
    C = out[4,:]
    X = C[2:end] .- C[1:(end-1)]
    nonpos = sum(X .<= 0)
    if nonpos > 0
        return Inf
    end
    -2.0*sum(logpdf.(Poisson.(X), data))
end;


lb = [0.0, 0.0]
ub = [1.0, 1.0]
θ = [0.01, 0.5]
θ₀ = [0.01, 0.1];


minustwologlik(θ₀)


res = Optim.optimize(minustwologlik, lb, ub, θ₀, Optim.Fminbox(NelderMead()))
res = Optim.optimize(minustwologlik, lb, ub, res.minimizer, Optim.Fminbox(LBFGS()))
θ̂ = res.minimizer


α = res.minimum + cquantile(Chisq(1), 0.05);


prof = Vector{ParamInterval}(undef,length(θ̂))
theta_bounds = [(lb[1],ub[1]),(lb[2],ub[2])]
eps = 1e-9
scan_bounds = [(0.0+eps,1.0-eps),(0.0+eps,1.0-eps)]
for i in eachindex(θ̂)
    prof[i] = get_interval(
        θ̂,
        i,
        minustwologlik,
        :CICO_ONE_PASS,
        loss_crit = α,
        theta_bounds = theta_bounds,
        scan_bounds = scan_bounds[i],
        scale = fill(:logit,length(θ̂))
    ) 
end;


ENV["COLUMNS"]=80
df_res = DataFrame(
    Parameters = [:I₀, :β], 
    StatusLower = [k.result[1].status for k in prof],
    StatusUpper = [k.result[2].status for k in prof],
    CILower = [k.result[1].value for k in prof],
    CIUpper = [k.result[2].value for k in prof],
    FittedValues = θ̂,
    NominalStartValues = θ₀
)
df_res


update_profile_points!.(prof);


l = @layout [a b]
p1 = plot(prof[1], xlabel="I₀", ylabel = "L(I₀)", legend=:top)
p2 = plot(prof[2], xlabel="β", ylabel = "L(β)", legend=:top)
plot(p1, p2, layout=l)


n_samples = 10000
lb2 = [k.result[1].value for k in prof] * 0.5
ub2 = [k.result[2].value for k in prof] * 2
lhs = QuasiMonteCarlo.sample(n_samples, lb2, ub2, LatinHypercubeSample());


lhs_result = [minustwologlik(lhs[:,i]) for i in 1:n_samples];


idx = (1:n_samples)[lhs_result .< α];


lhs_params = lhs[:,idx];


full_lowerci = minimum(lhs_params,dims=2)
full_upperci = maximum(lhs_params,dims=2);


ENV["COLUMNS"]=80
full_df_res = DataFrame(
    Parameters = [:I₀, :β],
    FullCILower = vec(full_lowerci),
    FullCIUpper = vec(full_upperci),
    ProfCILower = [k.result[1].value for k in prof],
    ProfCIUpper = [k.result[2].value for k in prof],
    FittedValues = θ̂,
    NominalStartValues = θ₀
)
full_df_res

