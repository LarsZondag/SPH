N = 2 # Number of particles
m = 1 # Mass of particles
ν = 2
σ = 2 / 3 * (ν == 1) + 10 / (7 * π) * (ν == 2) + 1 / π * (ν == 3)
c2 = 1
gamma = 1
h = 0.026
density_0 = 1
lx = 0.1*h
ly = 0.1*h
pos = rand((N, ν))
pos[:,1] *= lx
pos[:, 2] *= ly
vel = zeros((N, ν))
acc = zeros((N, ν))
dt = 0.004
N_t = 10


W_g(r, h, ν) = 1 / (h^ν * π^(ν/2)) * exp(- (r/h)^2)

function W_β(r, h, ν, σ)
    u = r / h
    if 0 <= u <= 1
      return σ / h^ν * (1 - 3 / 2 * u^2 + 3 / 4 * u^3)
    elseif 1 < u <= 2
      return σ * (2 - u)^3 / (4 * h^ν)
    else
      return 0
    end
end

function grad_W_β(r, h, ν, σ)
  d = sqrt(sum(r.^2))
  u = d / h
  if 0 <= u <= 1
    return σ / h^(ν + 2) * (9 / 4 * u^2 - 3 * u) * r[:]
  elseif 1 < u <= 2
    return σ / h^(ν + 2) * (- 3 / 4 * (2 - u)^2) * r[:]
  else
    return zeros(ν)
  end
end


function acceleration(positions)
  particles = length(positions[:, 1])
  density = ones(particles) * m * W_β(0, h, ν, σ)
  acceleration = zeros((particles, ν))

  distance = zeros((particles, particles, ν))
  for i = 1:particles
    for j = i+1:particles
      distance[i, j, :] = positions[i, :] - positions[j, :]
      d = sqrt(sum(distance[i, j, :].^2))
      densIncrement = m * W_β(d, h, ν, σ)
      density[i] += densIncrement
      density[j] += densIncrement
    end
  end

  pressure = density_0 * c2 / gamma * ((density/density_0)^gamma - 1)

  for i in 1:particles
    for j in i+1:particles
      acc_common = m * (pressure[i] / density[i]^2 + pressure[j] / density[j]^2) * grad_W_β(distance[i, j, :], h, ν, σ)
      acceleration[i, :] -= acc_common'
      acceleration[j, :] += acc_common'
    end
  end
  return acceleration
end

function make_time_step!(positions, velocities, accelerations)
  velocities += 0.5 * accelerations * dt
  positions += velocities * dt
  accelerations = acceleration(positions)
  velocities += 0.5 * accelerations * dt
end

println(pos)
for t in 1:N_t
  vel += 0.5 * acc * dt
  pos += vel * dt
  acc = acceleration(pos)
  vel += 0.5 * acc * dt
end
println(pos)
