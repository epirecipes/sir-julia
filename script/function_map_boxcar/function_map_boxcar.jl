
using Distributions
using DistributionsAD # enables AD rules for logpdf
using SimpleDiffEq
using Random
using Plots;


seed = 1234
Random.seed!(seed);


@inline function rate_to_proportion(r, Δt = 1.0)
    if r < 0 || isnan(r)
        return 0.0
    elseif isinf(r)
        return 1.0
    else
        result = 1 - exp(-r * Δt)
        # Ensure result is valid
        return isnan(result) ? 0.0 : max(min(result, 1.0), 0.0)
    end
end;


function hazard_vector(dist::Distribution{Univariate,Discrete}, K::Integer)
    @assert minimum(dist) == 0 "Minimum of distribution is not zero!"
    ks = 0:(K-1)

    # Unnormalised log-weights over the truncated support
    logw = logpdf.(Ref(dist), ks)  # stays AD-friendly

    # Normalise: logpmf = logw - logZ (stable log-sum-exp)
    m = maximum(logw)
    logZ = m + log(sum(exp.(logw .- m)))
    logpmf = logw .- logZ
    pmf = exp.(logpmf)

    # Survival: P(X ≥ i) for i = 0..K-1
    sf = reverse(cumsum(reverse(pmf)))

    T = eltype(pmf)
    default_h = T(0.01)
    isok = (sf .> zero(T)) .& isfinite.(sf)
    h = ifelse.(isok, pmf ./ sf, default_h)

    # Clamp to [0, 1]
    clamp.(h, zero(T), one(T))
end;


function truncated_geometric_mean(p::Real, K::Integer)
    @assert 0 < p < 1
    @assert K ≥ 1
    q = 1 - p
    num = q - K*q^K + (K-1)*q^(K+1)
    den = p * (1 - q^K)
    return num / den
end;


function sir_map!(du, u, p, t)
    (β, pγ, Δt) = p

    # Extract state variables
    (S, I, R) = u
    N = S + I + R     # total population
    
    # Calculate new infections
    λ = rate_to_proportion(β * I / N, Δt)
    new_inf = λ * S
    new_rec = pγ * I

    # Update
    du[1] = S - new_inf 
    du[2] = I + new_inf - new_rec
    du[3] = R + new_rec
    
    return nothing
end;


function sir_boxcar_map!(du, u, p, t)
    # Unpack parameters
    (β, h, K, Δt) = p
    # Extract state variables and process
    T = eltype(u)
    S = u[1]
    ΣI = sum(u[2:K+1])
    R = u[K+2]
    N = S + ΣI + R

    # Calculate new infections
    λ = rate_to_proportion(β * ΣI / N, Δt)
    new_inf = λ * S

    # Update S and start R accumulator
    du[1] = S - new_inf
    du[K+2] = R

    # Initialize infected classes for next step
    @inbounds begin
        du[2] = new_inf
        for k in 2:K
            du[k+1] = zero(T)
        end
    end

    # Progress cohorts
    @inbounds for k in 1:K
        current_I = u[k+1]
        leave = current_I * h[k]
        stay = current_I - leave
        if k < K
            du[k+2] += stay
        end
        du[K+2] += leave
    end

    return nothing
end;


tmax = 40
Δt = 1.0
nsteps = Int(tmax/Δt);


β = 0.5;


μ = 4.0    # Mean infectious period in standard exponential model
γ = 1.0/μ  # Recovery rate in standard exponential model
pγ = rate_to_proportion(γ, Δt)
p = (β, pγ, Δt);


mean(Exponential(μ)), mean(Geometric(pγ))*Δt


K = nsteps
d = Geometric(pγ)
h = hazard_vector(d, K)
p_boxcar = (β, h, K, Δt);


mean(d), truncated_geometric_mean(pγ,K)


ρ = 0.01
u0 = [1.0 - ρ, ρ, 0.0];


u0_boxcar = zeros(K+2)
u0_boxcar[1] = 1.0 - ρ
u0_boxcar[2] = ρ;


prob = DiscreteProblem(sir_map!, u0, (0, nsteps), p)
sol = solve(prob, SimpleFunctionMap())
S = sol[1, :]
ΣI = sol[2, :]
R = sol[3, :]
N = S .+ ΣI .+ R;


prob_boxcar = DiscreteProblem(sir_boxcar_map!, u0_boxcar, (0, nsteps), p_boxcar)
sol_boxcar = solve(prob_boxcar, SimpleFunctionMap())
S_boxcar = sol_boxcar[1, :]
ΣI_boxcar = sum.(eachcol(@view sol_boxcar[2:K+1, :]))
R_boxcar = sol_boxcar[K+2, :]
N_boxcar = S_boxcar .+ ΣI_boxcar .+ R_boxcar;


p_gvsbox = plot(0:Δt:tmax, [S, ΣI, R, N], 
             label=["S - Standard" "I - Standard" "R - Standard" "N - Standard"],
             color=[:blue :red :green], linewidth=2, linestyle=:solid,
             xlabel="Time Step", ylabel="Population",
             title="Population Dynamics: Standard vs Boxcar Method",
             legend=:topright, grid=true)

plot!(p_gvsbox, 0:Δt:tmax, [S_boxcar, ΣI_boxcar, R_boxcar, N_boxcar], 
      label=["S - Boxcar" "I - Boxcar" "R - Boxcar" "N - Boxcar"],
      color=[:lightblue :orange :lightgreen], linewidth=2, linestyle=:dash)

p_gvsbox


maximum([maximum(abs.(S .- S_boxcar)), maximum(abs.(ΣI .- ΣI_boxcar)), maximum(abs.(R .- R_boxcar))])


nb_r = 4   # Shape parameter
nb_p = (nb_r * pγ)/(1 - pγ + nb_r*pγ)
nb_d = NegativeBinomial(nb_r, nb_p)
h_nb = hazard_vector(nb_d, K)
p_boxcar_nb = (β, h_nb, K, Δt);


mean(d)*Δt, mean(nb_d)*Δt


std(d)*Δt, std(nb_d)*Δt


prob_boxcar_nb = DiscreteProblem(sir_boxcar_map!, u0_boxcar, (0, nsteps), p_boxcar_nb)
sol_boxcar_nb = solve(prob_boxcar_nb, SimpleFunctionMap())
S_boxcar_nb = sol_boxcar_nb[1, :]
ΣI_boxcar_nb = sum.(eachcol(@view sol_boxcar_nb[2:K+1, :]))
R_boxcar_nb = sol_boxcar_nb[K+2, :];


p_gvsnb = plot(0:Δt:tmax, [S_boxcar, ΣI_boxcar, R_boxcar], 
             label=["S - G" "I - G" "R - G"],
             color=[:blue :red :green], linewidth=2, linestyle=:solid,
             xlabel="Time Step", ylabel="Population",
             title="Population Dynamics: Geometric Boxcar vs NB Boxcar",
             legend=:topright, grid=true)

plot!(p_gvsnb, 0:Δt:tmax, [S_boxcar_nb, ΣI_boxcar_nb, R_boxcar_nb], 
      label=["S - NB" "I - NB" "R - NB"],
      color=[:blue :red :green], linewidth=2, linestyle=:dash)

p_gvsnb

