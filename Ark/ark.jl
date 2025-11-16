using Ark, Plots

# "parameters"
N::Int = 1_000
I0::Int = 5
tmax::Float64 = 100
dt::Float64 = 0.1
steps::Int = Int(floor(tmax/dt))
gamma::Float64 = 1/10
R0::Float64 = 2.5
beta::Float64 = R0 * gamma

abstract type HealthState end
struct S <: HealthState end
struct I <: HealthState end
struct R <: HealthState end

world = World(S,I,R)

for _ in 1:N-I0
    new_entity!(world, (S(),))
end

for _ in 1:I0
    new_entity!(world, (I(),))
end

function get_count(world, ::Type{T}) where {T<:HealthState}
    count = 0
    for (entities, _) in @Query(world, (T, ))
        count += length(entities)
    end
    return count
end

# output
trajectory = zeros(Int, steps+1, length(subtypes(HealthState)))
trajectory[1, :] = [get_count(world, T) for T in (S, I, R)]

# run sim
for t in 1:steps
    # S->I
    i = get_count(world, I)
    foi = beta * i/N
    prob = 1-exp(-foi*dt)
    s_to_i = Entity[]
    for (entities, _) in @Query(world, (S, ))
        @inbounds for i in eachindex(entities)
            if rand() <= prob
                push!(s_to_i, entities[i])
            end
        end
    end
    # I->R
    prob = 1-exp(-gamma*dt)
    i_to_r = Entity[]
    for (entities, _) in @Query(world, (I, ))
        @inbounds for i in eachindex(entities)
            if rand() <= prob
                push!(i_to_r, entities[i])
            end
        end
    end
    # apply transitions
    for entity in s_to_i
        @exchange_components!(world, entity, 
            add    = (I(),),
            remove = (S, ),
        )
    end
    for entity in i_to_r
        @exchange_components!(world, entity, 
            add    = (R(),),
            remove = (I, ),
        )
    end
    # record state
    trajectory[t+1, :] = [get_count(world, T) for T in (S, I, R)]
end

plot(
    trajectory,
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number"
)