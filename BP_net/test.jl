using Flux
1+1
# %%
W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1

W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2 * x .+ b2

model(x) = layer2(σ.(layer1(x)))
model(rand(5))
# %%
function linear(in, out)
  W = randn(out, in)
  b = randn(out)
  x -> W * x .+ b
end
linear1 = linear(5, 3)
linear2 = linear(3, 2)
model(x) = linear2(σ.(linear1(x)))
model(rand(5))
# %%
struct Affine
  W
  b
end
Affine(in::Integer, out::Integer) =
  Affine(randn(out, in), randn(out))
(m::Affine)(x) = m.W * x .+ m.b
a = Affine(10, 2)
a(rand(10))

# %%

layers1 = Dense(10, 5, σ)
layers = [Dense(10, 5, σ), Dense(5, 2), softmax]
model(x) = foldl((x, m) -> m(x), layers, init = x)
model(rand(10))
# %%
model2 = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)
model2(rand(10))
# %%
W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

grads = gradient(() -> loss(x, y), params([W, b]))
# %%
using Flux.Optimise: update!
η = 0.1
for p in (W, b)
  update!(p, -η * grads[p])
end
