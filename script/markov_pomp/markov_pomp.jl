
using PartiallyObservedMarkovProcesses
using Distributions
using DataFrames
using Loess
using Random
using Plots
using Base.Threads;


Random.seed!(1234); # For reproducibility


sir_rinit = function (;S₀,I₀,R₀,_...)
    return (S=S₀, I=I₀, R=R₀, C=0)
end;


sir_rprocess = function(;t,S,I,R,C,β,γ,N,dt,_...)
    infprob = 1-exp(-β*I/N*dt)
    recprob = 1-exp(-γ*dt)
    infection = rand(Binomial(S,infprob))
    recovery = rand(Binomial(I,recprob))
    return (S=S-infection,
            I=I+infection-recovery,
            R=R+recovery,
            C=C+recovery,
            )
end;


sir_rmeasure_exact = function (;C,_...)
    return (Y=C,)
end

sir_logdmeasure_exact = function (;Y,C,_...)
    if Y==C
        return 0.0
    else
        return -Inf
    end
end;


sir_rmeasure_underreport = function (;C,q,_...)
    return (Y=rand(Binomial(C,q)),)
end

sir_logdmeasure_underreport = function (;Y,C,q,_...)
    return logpdf(Binomial(C,q),Y)
end;


p = (β = 0.5, # Infectivity
     γ = 0.25, # Recovery rate
     q = 0.75, # Fraction of new cases observed
     N  = 1000.0, # Total population size (as a float)
     S₀ = 990, # Initial susceptibles
     I₀ = 10, # Initial infected
     R₀ = 0); # Initial recovered


t₀ = 0.0
δt = 1.0
times = collect(0:1.0:40.0);


s = simulate(
        params = p,
        t0 = t₀,
        times = times,
        accumvars = (C=0,),
        rinit = sir_rinit,
        rprocess = euler(sir_rprocess, dt = δt),
        rmeasure = sir_rmeasure_underreport,
        logdmeasure =  sir_logdmeasure_underreport
    )[1];


time_vec = s.times
st = states(s);


S_vec, I_vec, R_vec, C_vec = [getproperty.(st, s) for s in (:S, :I, :R, :C)];


Y_vec = getproperty.(obs(s), :Y)
dat = DataFrame(time = time_vec, Y=Y_vec);


ZN = 100
Z = rand(Binomial(ZN,R_vec[end]/1000.0));


plot(times,
    [S_vec I_vec R_vec],
    line = :path,
    marker = (:circle, 3),
    labels = ["S" "I" "R"])


plot(times,
    [C_vec Y_vec],
    line = :path,
    marker = (:circle, 3),
    labels = ["C" "Y"])


P = pomp(dat;
         times=:time,
         t0=t₀,
         rinit=sir_rinit,
         rprocess=euler(sir_rprocess, dt = δt),
         rmeasure=sir_rmeasure_underreport,
         logdmeasure=sir_logdmeasure_underreport,
         params=p,
         accumvars=(C=0,)
);


Pf = pfilter(P, Np=1000, params=p)
println(
    "PartiallyObservedMarkovProcesses.jl likelihood estimate: ",
    round(Pf.logLik,digits=2)
)


betas = collect(0.35:0.005:0.65)
nbetas = length(betas)
beta_liks = Array{Float64}(undef,nbetas)
Threads.@threads for i in 1:nbetas
    pc = merge(p, (β=betas[i],))
    beta_liks[i] = pfilter(P, Np=10000, params=pc).logLik
end

betas_model = loess(betas, beta_liks)
beta_liks_smooth = Loess.predict(betas_model, betas)
β̂=betas[argmax(beta_liks_smooth)]
plot(betas,
    beta_liks_smooth,
    xlabel="β",
    ylabel="Log likelihood",
    label="",
    legend=true,
    marker=false)
scatter!(betas, beta_liks, label="")
vline!([p.β],label="True β")
vline!([β̂],label="Estimated β")


I0s = collect(5:1:50)
nI0s = length(I0s)
I0_liks = Array{Float64}(undef,nI0s)
Threads.@threads for i in 1:nI0s
    pc = merge(p, (I₀=I0s[i],))
    I0_liks[i] = pfilter(P, Np=10000, params=pc).logLik
end

I0s_model = loess(I0s, I0_liks)
I0_liks_smooth = Loess.predict(I0s_model, I0s)
Î₀=I0s[argmax(I0_liks_smooth)]
plot(I0s,
    I0_liks_smooth,
    xlabel="I₀",
    ylabel="Log likelihood",
    label="",
    legend=true,
    marker=false)
scatter!(I0s, I0_liks, label="")
vline!([p.I₀],label="True I₀")
vline!([Î₀],label="Estimated I₀")


using Turing
using MCMCChains
using StatsPlots;


@model function sir_particle_mcmc_fixed_q(P)
    # Priors for the parameters we want to estimate
    β ~ Uniform(0.1, 0.9)
    I₀ ~ DiscreteUniform(5, 50)
    
    # Create parameter tuple with current MCMC values
    current_params = merge(P.params, (β=β, I₀=I₀))

    # Compute particle filter likelihood
    pf_result = pfilter(P, Np=1000, params=current_params)  # Reduced particles for speed
    
    # Add the log-likelihood to the model
    Turing.@addlogprob! pf_result.logLik
    
    return nothing
end;


sir_model_fixed_q = sir_particle_mcmc_fixed_q(P);


n_samples = 11000
n_chains = 2
chain_fixed_q = sample(sir_model_fixed_q,
                          MH(),
                          MCMCThreads(),
                          n_samples,
                          n_chains;
                          progress=false);


describe(chain_fixed_q)


plot(chain_fixed_q)


@model function sir_particle_mcmc_incorrect_q(P)
    # Priors for the parameters we want to estimate
    β ~ Uniform(0.1, 0.9)
    I₀ ~ DiscreteUniform(5, 50)
    
    # Create parameter tuple with current MCMC values
    current_params = merge(P.params, (β=β, I₀=I₀, q=1.0))

    # Compute particle filter likelihood
    pf_result = pfilter(P, Np=1000, params=current_params)  # Reduced particles for speed
    
    # Add the log-likelihood to the model
    Turing.@addlogprob! pf_result.logLik
    
    return nothing
end;


sir_model_incorrect_q = sir_particle_mcmc_incorrect_q(P);


n_samples = 11000
n_chains = 2
chain_incorrect_q = sample(sir_model_incorrect_q,
                           MH(),
                           MCMCThreads(),
                           n_samples,
                           n_chains;
                           progress=false);


describe(chain_incorrect_q)


plot(chain_incorrect_q)


@model function sir_particle_mcmc_estimate_q(P)
    # Priors for the parameters we want to estimate
    β ~ Uniform(0.1, 0.9)
    I₀ ~ DiscreteUniform(5, 50)
    q ~ Uniform(0.25, 0.75)

    # Create parameter tuple with current MCMC values
    current_params = merge(P.params, (β=β, I₀=I₀, q=q))

    # Compute particle filter likelihood
    pf_result = pfilter(P, Np=1000, params=current_params)  # Reduced particles for speed
    
    # Add the log-likelihood to the model
    Turing.@addlogprob! pf_result.logLik
    
    return nothing
end;


sir_model_estimate_q = sir_particle_mcmc_estimate_q(P);


n_samples = 11000
n_chains = 2
chain_estimate_q = sample(sir_model_estimate_q,
                           MH(),
                           MCMCThreads(),
                           n_samples,
                           n_chains;
                           progress=false);


describe(chain_estimate_q)


plot(chain_estimate_q)


@model function sir_particle_mcmc_estimate_q_prevalence(P, Z, ZN)
    # Priors for the parameters we want to estimate
    β ~ Uniform(0.1, 0.9)
    I₀ ~ DiscreteUniform(5, 50)
    q ~ Uniform(0.25, 0.75)
    
    # Create parameter tuple with current MCMC values
    current_params = merge(P.params, (β=β, I₀=I₀, q=q))

    # Compute particle filter likelihood
    pf_result = pfilter(P, Np=1000, params=current_params)  # Reduced particles for speed

    # Calculate contribution from end prevalence study
    zp = pf_result.traj[end][:R]/1000.0
    zp = max(min(zp,1.0),0.0) # To ensure boundedness
    Z ~ Binomial(ZN, zp)

    # Add the log-likelihood to the model
    Turing.@addlogprob! pf_result.logLik
    
    return nothing
end;


sir_model_estimate_q_prevalence = sir_particle_mcmc_estimate_q_prevalence(P, Z, ZN);


n_samples = 11000
n_chains = 2
chain_estimate_q_prevalence = sample(sir_model_estimate_q_prevalence,
                                     MH(),
                                     MCMCThreads(),
                                     n_samples,
                                     n_chains;
                                     progress=false);


describe(chain_estimate_q_prevalence)


plot(chain_estimate_q_prevalence)

