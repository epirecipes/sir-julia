using Pkg

unregistered = [
   ("https://github.com/sdwfrost/Gillespie.jl","master"),
   ("https://github.com/augustinas1/MomentClosure.jl","main")
]

for u in unregistered
  Pkg.add(PackageSpec(url=u[1], rev=u[2]))
end
