using Ark, Random, Plots, BenchmarkTools;

function rate_to_probability(r::T, t::T) where {T<:AbstractFloat}
    1 - exp(-r * t)
end;

# Time domain parameters
δt = 0.1
nsteps = 400
tf = nsteps * δt
t = 0:δt:tf;

# System parameters
β = 0.05
c = 10.0
γ = rate_to_probability(0.25, δt);

# Initial conditions
N = 1_000
I0 = 5;

abstract type HealthState end
struct S <: HealthState end
struct I <: HealthState end
struct R <: HealthState end;

function get_count(world, ::Type{T}) where {T<:HealthState}
	count = 0
	for (entities, ) in @Query(world, (), with=(T, ))
		count += length(entities)
	end
	return count
end;

function run_sir()

	# set up world, add initial entities
	world = World(S,I,R)
	new_entities!(world, N-I0, (S(),))
	new_entities!(world, I0, (I(),))

	# output
	trajectory = zeros(Int, nsteps+1, length(subtypes(HealthState)))
	trajectory[1, :] = [get_count(world, T) for T in (S, I, R)]

	# vectors to contain Entities making state transitions on this step
	i_to_r = Entity[]
	s_to_i = Entity[]

	# run sim
	for t in 1:nsteps
	    # S->I
	    i = get_count(world, I)
	    foi = β * c * i/N
	    prob = rate_to_probability(foi, δt)
	    for (entities, ) in @Query(world, (), with=(S, ))
	        @inbounds for i in eachindex(entities)
	            if rand() <= prob
	                push!(s_to_i, entities[i])
	            end
	        end
	    end
	    # I->R
	    for (entities, ) in @Query(world, (), with=(I, ))
	        @inbounds for i in eachindex(entities)
	            if rand() <= γ
	                push!(i_to_r, entities[i])
	            end
	        end
	    end
	    # apply transitions
	    for entity in s_to_i
	        @exchange_components!(world, entity, add = (I(),), remove = (S,))
	    end
	    for entity in i_to_r
	        @exchange_components!(world, entity, add = (R(),), remove = (I,))
	    end
	    # record state
	    trajectory[t+1, :] = [get_count(world, T) for T in (S, I, R)]
        # reuse vectors to avoid allocations
	    resize!(i_to_r, 0)
	    resize!(s_to_i, 0)
	end
	
	return trajectory
end;

trajectory = run_sir()

plot(
    t,
    trajectory,
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number"
)

@benchmark run_sir()
