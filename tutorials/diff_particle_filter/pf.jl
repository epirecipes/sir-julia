
function particle_filter(data, m::Integer, x0::T, θ; store_path = false) where {T <: SVector}
    X = fill(x0, m)
    W = [1/m for i in 1:m]
    ω = sum(W) # total weight
    
    n = 1 # timestep of particles

    store_path && (Xs = [X])
    
    for i in axes(data,1)
        t = data[i,1]
        # propagate particles to next data time point
        while n < t
            # update all particles
            for j in 1:m
                X[j] = dyn(X[j],θ)
            end
            n += 1
        end
        store_path && Zygote.ignore(() -> push!(Xs, X))
        # update weights
        wi = map((x) -> pdf(Poisson(x[2]), data[i,2]), X)
        W = W .* wi
        ω = sum(W)
        # resample particles
        if t < size(data,1)
            X, W = resample(m, X, W, ω, sample_stratified)
        end
    end
    
    return (store_path ? Xs : X), W
end

