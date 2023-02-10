using MeasureTheory
using Plots

# mc = Chain(x -> Normal(μ=x), Normal(μ=0.0))
mc = Chain(x -> Normal(μ=x), Dirac(0.0))
r = rand(mc)
samp = Iterators.take(r, 1000)
plot(collect(samp))


function counting(x)
    # [Dirac(x[1]+1), Dirac(x[2]-1)]
    Dirac([x[1]+1, x[2]-1])
end

# mc = Chain(x -> counting(x), [Dirac(1),Dirac(2)])
mc = Chain(x -> counting(x), Dirac([0,0]))
r = rand(mc)
samp = Iterators.take(r, 1000)
plot(transpose(hcat(collect(samp)...)))