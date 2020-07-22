using CSV,DataFrames,PyCall,Statistics,RCall,JLD
using Flux
using Flux: onehotbatch,logitcrossentropy,onecold,@epochs,RNN,GRU,chunk
using Flux.Data: DataLoader
using Parameters: @with_kw

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end


#读取数据
@load "/home/jerrywang/Desktop/var.jld"

#调参
@with_kw mutable struct args
    η::Float64 = 3e-4     # 学习率
    batchsize::Int = 50   # 批量处理的数量   batch size
    epochs::Int = 10       # 训练次数
    device::Function = gpu  # set as gpu, if gpu available
end
Arg=args()

## 输出结果编码


train_out,test_out=chunk(onehotbatch(train[:,3],-1:1),size(train,1)),chunk(onehotbatch(test[:,3],-1:1),size(test,1))
train_input,test_input=train[:,2],test[:,2]

train_data=DataLoader(train_input,train_out,batchsize=Arg.batchsize, shuffle=true)
test_data=DataLoader(test_input,test_out,batchsize=Arg.batchsize, shuffle=true)

## GRU网络训练
# 模型训练参数调整

# 建立模型
m = Chain(GRU(1, 64),Dense(64,3))

# 
function eval_model(x)
    out = m(x)[end]
    Flux.reset!(m)
    out
end
# 训练
loss(x, y) = logitcrossentropy(eval_model(x) ,y)
println("训练前损失为 = ", sum(loss.(train_input, train_out)))
opt = ADAM(0.001)


function accuracy(data_loader)
    l=0
    for (i,o) in data_loader
        l+=(onecold(eval_model(i))[1] == onecold(o))
    end
    l/size(x,1)
end

evalcb() = @show(sum(loss.(train_input, train_out)))
@epochs Arg.epochs Flux.train!(loss, params(m), zip(train_input,train_out), opt, cb = Flux.throttle(evalcb,1))
@show accuracy(test_input, test_out)
