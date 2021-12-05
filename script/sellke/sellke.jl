
using Distributions
using Random
using BenchmarkTools
using Plots


function sellke(u0, p)
    (S, I, R) = u0
    N = S + I + R
    (β, c, γ) = p
    λ = β*c/N

    Q = rand(Exponential(), S)
    sort!(Q)

    T0 = rand(Exponential(1/γ), I)
    T = rand(Exponential(1/γ), S)

    ST0 = sum(T0)
    Y = [ST0; ST0 .+ cumsum(T[1:end-1])]

    Z = findfirst(Q .> Y*λ)

    if isnothing(Z)
        Z = S + I # entire population infected
    else
        Z = Z + I - 1
    end

    TT = [T0; T] # all infectious periods
    QQ = [T0 * 0; Q] # all thresholds 
    R = T[1:I] # recovery times of the initial infectives

    # max num of events possible
    max = I + 2*S

    t = zeros(max)
    St = zeros(max)
    It = zeros(max)
    It[1:I] = (1:I)
    St[1:I] = N .- (1:I)

    tt = 0
    La = 0
    j = I+1
    k = j

    while It[k-1] > 0
        (minR, i) = findmin(R)
        dtprop = minR-tt
        Laprop = La + (λ * It[k-1] * dtprop)
        if j > length(QQ) # only recoveries remain
            R = R[setdiff(1:length(R), i)]
            tt = minR
            t[k] = minR
            It[k] = It[k-1]-1
            St[k] = St[k-1]
            La = Laprop
            k = k+1
        else # infections
            if QQ[j] > Laprop
                R = R[setdiff(1:length(R), i)]
                tt = minR
                t[k] = minR
                It[k] = It[k-1]-1
                St[k] = St[k-1]
                La = Laprop
            else
                tt = tt + ((QQ[j]-La)/(Laprop-La))*dtprop
                La = QQ[j]
                t[k] = tt
                It[k] = It[k-1]+1
                St[k] = St[k-1]-1
                R = [R; tt+TT[j]]
                j = j+1
            end
            k = k+1
        end
    end
    
    trajectory = hcat(t, St, It, N .- (St + It))
    trajectory = trajectory[I:k-1, :]

    return trajectory
end


u0 = [990,10,0]; # S,I,R


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


out = sellke(u0, p)


plot(out[:, 1], out[:, 2:end],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark sellke(u0, p)

