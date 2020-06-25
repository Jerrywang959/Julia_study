using ClassImbalance, CSV, Flux, MLDataPattern
using Flux,DataFrames,CSV,Random
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using Random: MersenneTwister

# 参数调整
@with_kw mutable struct args
    η::Float64 = 3e-4     # 学习率
    batchsize::Int = 5   # 批量处理的数量   batch size
    epochs::Int = 10       # 训练次数
    device::Function = cpu  # set as gpu, if gpu available
end

## 预处理数据
1+1
# 建立模型
function build_model()
    return (Chain(Dense(3,24,relu),Dense(24,24,tanh),Dense(24,2),softmax))
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += Flux.logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

function train(args)
    # Initializing Model parameters 

    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
    train_data = args().device.(train_data)
    test_data = args().device.(test_data)
    m = args().device(m)
    loss(x,y) = logitcrossentropy(m(x), y)
    
    ## Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args().η)
		
    @epochs args().epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)

    @show accuracy(test_data, m)

    return m
end

cd(@__DIR__)
train(args)


