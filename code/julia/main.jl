N = 10 # Number of particles
m = 1 # Mass of particles
ν = 2
lx = 5
ly = 2
pos = rand((N, ν))
pos[:,1] *= lx
pos[:, 2] *= ly
vel = zeros((N, ν))
σ = 2 / 3 * (ν == 1) + 10 / (7 * π) * (ν == 2) + 1 / π * (ν == 3)
c2 = 1
gamma = 1
h = 0.026
density_0 = 1


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

function ∇W_β(r, h, ν, σ)
    u = r / h
    if 0 <= u <= 1
      return σ / h^(ν + 1) * (9 / 4 * u^2 - 3 * u)
    elseif 1 < u <= 2
      return σ / h^(ν + 1) * (- 3 / 4 * (2 - u)^ 2)
    else
      return 0
    end
end


function acceleration(positions)
  particles = length(positions)
  density = zeros(particles)
  acceleration = zeros((particles, ν))
  for i = 1:particles
    for j = i+1:particles
      distance = positions[i] - positions[j]
      d = sum(distance^2)
      densIncrement = m * W_β(d, h, ν, σ)
      density[i] += densIncrement
      density[j] += densIncrement
    end
  end

  pressure = density_0 * c2 / gamma * ((density/density_0)^gamma - 1)

  return density
end

acceleration(pos)
