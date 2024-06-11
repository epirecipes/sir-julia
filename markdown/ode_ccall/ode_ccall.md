# Ordinary differential equation model with the vector field defined in C
Simon Frost (@sdwfrost), 2024-06-03

## Introduction

While Julia is a high-level language, it is possible to define the vector field for an ordinary differential equation (ODE) in C and call it from Julia. This can be useful for performance reasons (if the calculation of the vector field in Julia happens to be slow), or if the vector field is already defined in C, for example, in another codebase. Julia's `ccall` makes it easy to call a compiled C function in a shared library.

## Libraries

```julia
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools
```




## Transitions

We define the vector field in C; it is easiest for this function to be in-place, so that we do not have to do any memory management on the C side. This approach is also more efficient, as it reduces the number of allocations needed. Note that, as in Julia, we can use Unicode characters for the parameters, making the code look more similar to the mathematical equations.

```julia
C_code = """
void sir_ode(double *du, double *u, double *p, double *t){
    double β = p[0];
    double c = p[1];
    double γ = p[2];
    double S = u[0];
    double I = u[1];
    double R = u[2];
    double N = S + I + R;
    du[0] = -β*c*S*I/N;
    du[1] = β*c*S*I/N - γ*I;
    du[2] = γ*I;
}
""";
```




We then compile the code into a shared library; this example uses `gcc`, but any C compiler should work (though the invocation will be different). We use `tempname` to create a temporary file name for the shared library; actually, this will be the filename without the extension, as we will add the extension later, as the extension is platform-dependent.

```julia
const Clib = tempname();
```




`gcc` allows shared libraries to be generated directly via piping the C code to the compiler. Note the use of the `Libdl.jl` package, which will provide the platform-dependent extension for shared libraries.

```julia
open(`gcc -fPIC -O3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`, "w") do f
    print(f, C_code)
end;
```



A simpler approach is to just save the C code to a file and then compile it.

```julia
open(Clib * "." * "c", "w") do f
    write(f, C_code)
end
run(`gcc -fPIC -O3 -shared -o $(Clib * "." * Libdl.dlext) $(Clib * "." * "c")`);
```




We can then define the ODE function in Julia, which calls the C function using `ccall`. `du`, `u`, `p` are arrays of `Float64`, which are passed using pointers. `t` is passed as a `Ref` to a `Float64`, which is a pointer to a `Float64` value.

```julia
function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Clib,), Cvoid,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), du, u, p, Ref(t))
end;
```




We can then proceed to solve the ODE using the `sir_ode_jl!` function as we would if the vector field were defined in Julia.

```julia
sir_ode = ODEFunction(sir_ode_jl!);
```




## Time domain

```julia
δt = 0.1
tmax = 40.0
tspan = (0.0,tmax);
```




## Initial conditions and parameter values

```julia
u0 = [990.0,10.0,0.0] # S,I,R
p = [0.05,10.0,0.25]; # β,c,γ
```




## Solving the ODE

```julia
prob_ode = ODEProblem{true}(sir_ode, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), dt = δt);
```




## Plotting

```julia
plot(sol_ode, labels=["S" "I" "R"], lw = 2, xlabel = "Time", ylabel = "Number")
```

![](figures/ode_ccall_11_1.png)



## Benchmarking

```julia
@benchmark solve(prob_ode, Tsit5(), dt = δt)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   8.125 μs …   6.567 ms  ┊ GC (min … max):  0.00% … 99.
24%
 Time  (median):      9.791 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   12.043 μs ± 110.188 μs  ┊ GC (mean ± σ):  17.15% ±  1.
98%

            ▄▂▇▃██▂▅ ▃                                          
  ▁▁▁▂▂▂▃▅▇▇██████████▆▇▇▄▅▃▃▂▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  8.12 μs         Histogram: frequency by time         14.5 μs <

 Memory estimate: 14.69 KiB, allocs estimate: 165.
```





We can compare the performance of the C-based ODE with the Julia-based ODE.

```julia
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
end
prob_ode2 = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode2 = solve(prob_ode2, Tsit5(), dt = δt)
@benchmark solve(prob_ode2, Tsit5(), dt = δt)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   8.167 μs …  2.286 ms  ┊ GC (min … max): 0.00% … 98.22
%
 Time  (median):     10.000 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.831 μs ± 39.083 μs  ┊ GC (mean ± σ):  6.17% ±  1.70
%

           ▁ ▃▁▆▃█▃▇▂▅ ▃ ▂                                     
  ▁▁▁▁▂▃▃▅▄█▇███████████▇█▅▇▄▅▃▄▂▃▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  8.17 μs         Histogram: frequency by time        14.3 μs <

 Memory estimate: 15.08 KiB, allocs estimate: 173.
```





Note that the performance of the C-based vector field is similar to the one defined in Julia.