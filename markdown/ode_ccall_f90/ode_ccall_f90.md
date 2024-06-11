# Ordinary differential equation model with the vector field defined in Fortran 90
Simon Frost (@sdwfrost), 2024-06-03

## Introduction

While Julia is a high-level language, it is possible to define the vector field for an ordinary differential equation (ODE) in Fortran and call it from Julia. This can be useful for performance reasons (if the calculation of the vector field in Julia happens to be slow), or if the vector field is already defined in Fortran, for example, in another codebase. Julia's `ccall` makes it easy to call a compiled Fortran function in a shared library.

## Libraries

```julia
using OrdinaryDiffEq
using Libdl
using Plots
using BenchmarkTools
```




## Transitions

We define the vector field in Fortran 90; it is easiest for this function to be in-place, so that we do not have to do any memory management on the Fortran side. This approach is also more efficient, as it reduces the number of allocations needed. We use the `bind(c)` attribute to ensure that the function is callable from C.

```julia
F90_code = """
module sir_module
    use iso_c_binding

    contains

    subroutine sir_ode(du, u, p, t) bind(c)
        real(c_double), intent(out) :: du(3)
        real(c_double), intent(in) :: u(3)
        real(c_double), intent(in) :: p(3)
        real(c_double), intent(in) :: t

        real(c_double) :: beta, c, gamma, S, I, R, N

        beta = p(1)
        c = p(2)
        gamma = p(3)
        S = u(1)
        I = u(2)
        R = u(3)
        N = S + I + R

        du(1) = -beta*c*S*I/N
        du(2) = beta*c*S*I/N - gamma*I
        du(3) = gamma*I
    end subroutine sir_ode

end module sir_module
""";
```




We then compile the code into a shared library using `gfortran`. We use `tempname` to create a temporary file name for the shared library; actually, this will be the filename without the extension, as we will add the extension later, as the extension is platform-dependent.

```julia
const Flib = tempname();
```




We save the F90 code to a file and then compile it.

```julia
open(Flib * "." * "f90", "w") do f
    write(f, F90_code)
end
run(`gfortran -fPIC -shared -O3 -o $(Flib * "." * Libdl.dlext) $(Flib * "." * "f90")`);
```




We can then define the ODE function in Julia, which calls the F90 function using `ccall`. `du`, `u`, `p` are arrays of `Float64`, which are passed using pointers. `t` is passed as a `Ref` to a `Float64`, which is a pointer to a `Float64` value.

```julia
function sir_ode_jl!(du,u,p,t)
    ccall((:sir_ode,Flib,), Cvoid,
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

![](figures/ode_ccall_f90_10_1.png)



## Benchmarking

```julia
@benchmark solve(prob_ode, Tsit5(), dt = δt)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   8.125 μs …   6.705 ms  ┊ GC (min … max):  0.00% … 99.
41%
 Time  (median):      9.958 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   12.217 μs ± 112.451 μs  ┊ GC (mean ± σ):  17.24% ±  1.
98%

            ▂ ▅▂▄▆▃█▃█▂▃ ▁                                      
  ▁▁▁▁▂▂▃▃▆▆█▇██████████▇█▆▇▄▅▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  8.12 μs         Histogram: frequency by time         14.3 μs <

 Memory estimate: 14.69 KiB, allocs estimate: 165.
```





We can compare the performance of the F90-based ODE with the Julia-based ODE.

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
 Range (min … max):   8.208 μs …  2.296 ms  ┊ GC (min … max): 0.00% … 98.38
%
 Time  (median):     10.250 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   11.099 μs ± 38.851 μs  ┊ GC (mean ± σ):  5.99% ±  1.70
%

               ▃▂▆▃▃█▃▇▂▆▁▃  ▁                                 
  ▂▁▂▂▂▂▃▃▃▆▅█▇█████████████▇█▆▇▅▆▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▄
  8.21 μs         Histogram: frequency by time        14.2 μs <

 Memory estimate: 15.08 KiB, allocs estimate: 173.
```





Note that the performance of the F90-based vector field is similar to the one defined in Julia.