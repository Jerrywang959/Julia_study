using Flux
using Flux:onehotbatch,logitcrossentropy,RNN,update!,reset!
x =[randn(50) for i=1:10]
yy=[-1,1,0,1,1,0,0,1,-1,1]
y=Flux.chunk(onehotbatch(yy,-1:1),10)
m =RNN(1, 3)
opt = ADAM(0.001)
function eval_model(x)
    out = m.(x)[end]
    Flux.reset!(m)
    out
end
loss(x, y) = logitcrossentropy(eval_model(x) ,y)
println("Training loss before = ", sum(loss.(x, y)))
evalcb() = @show(sum(loss.(x, y)))
Flux.train!(loss, params(m),zip(x,y), opt, cb = Flux.throttle(evalcb,1))