using CSV,DataFrames,PyCall,Statistics,RCall,JLD
using Flux
using Flux: onehotbatch,logitcrossentropy,onecold,@epochs,RNN,GRU,chunk
#using Flux.Data: DataLoader
using Parameters: @with_kw


## 导入python模型TLP
ltp=pyimport("ltp").LTP()

## 转化python模型为函数,方便输出,其中输入为字符串,
function getword(x::String)::Array{String,1}
    seg,_=ltp.seg([x])
    return seg[:]
end

## 读取数据
Trainraw=CSV.read("/home/jerrywang/Desktop/Final_project.csv")
aim=Trainraw[!,:sentiment_value]


## 分词

Wordset=by(Trainraw ,:content_id  , word = :content => x->getword.(x) )

## 删除标点符号和数字
point=r"？*，*。*：*,*:*\?*\.*"    # 标点符号的正则表达式
## 写函数批量处理
function deletepoint(x::Array{String,1})::Array{String,1}
    de=Int64[]
    y=copy(x)     # copy副本,以免更改之前的变量
    for i in keys(x)
        value=x[i]
        if typeof(value)==Char
            value="$value"
        end
        if match(point,value).match!=""
            push!(de,i)
        end
    end
    deleteat!(y,de)
end
# 正式删除标点符号
PureWord=by(Wordset,:content_id,pure_word=:word => x-> deletepoint.(x) )

## 分词后统一长度
# 获取最长的分词长度
max_length=maximum(length.(PureWord[!,2]))


# 补全函数
function complete(x::Array{String,1})::Array{String,1}
    # 生成一个补全模板
    tem=Array{String,1}(undef,max_length)
    fill!(tem,"")     # 把空缺的换成语料库中查不到的元素,以方便后面的替换
    tem[keys(x)].=x
    return tem
end
PureWord_lengthed=by(PureWord,:content_id, wordlengthed=:pure_word=>x->complete.(x) )

## 分词向量化
# 预先训练的模型来自 https://github.com/Embedding/Chinese-Word-Vectors, 选择从微博微博语料库预训练的模型
# 读取模型   
model=readlines("/home/jerrywang/Desktop/sgns.weibo.bigram-char")[2:end]
modelDict=Dict{String,Array{Float64,1}}()
# 注: 此步骤并行存在问题,还是多用矩阵运算比较好
for i in model
    u=split(i," ")
    modelDict[u[1]]=parse.(Float64,u[2:end-1])
end

# 一个词语转化为向量  不在语料库里则转为0
word2vec(x::String)::Array{Float64,1}=x in keys(modelDict) ? modelDict[x] : zeros(Float64,300)
# 词向量取平均生成一个句向量
sentence2vec(x::Array{String,1})::Array{Float64,1}=map(x->mean(word2vec(x)),x)
# 完成转换
WordVector=by(PureWord_lengthed,:content_id, vec=:wordlengthed=>x->sentence2vec.(x))
WordVector.y=aim


## R 分割数据集
@rput WordVector

R" library(caTools)"
R"""
set.seed(3000)
split = sample.split(WordVector$y, SplitRatio = 0.7)
train = subset(WordVector, split == TRUE)
test = subset(WordVector, split == FALSE)
"""
# 从R中取出数据
@rget train
@rget test

#暂存结果,方便下次使用
@save "/home/jerrywang/Desktop/var.jld" train test
@load "/home/jerrywang/Desktop/var.jld"

## 输出结果编码


train_out,test_out=chunk(onehotbatch(train[:,3],-1:1),size(train,1)),chunk(onehotbatch(test[:,3],-1:1),size(test,1))
train_input,test_input=train[:,2],test[:,2]

#train_input,test_input=zeros(max_length,size(train,1)),zeros(max_length,size(test,1))
#for i in 1:size(train,1)
#    train_input[:,i]=train[:,2][i]
#end
#for i in 1:size(test,1)
#    test_input[:,i]=test[:,2][i]
#end

#  数据放入DataLoader
#train_data = DataLoader(train_input, train_out, batchsize=args().batchsize, shuffle=true)
#test_data = DataLoader(test_input, test_out, batchsize=args().batchsize)





#function loss_all(dataloader, model)
#    l = 0f0
#    for (x,y) in dataloader
#        l += Flux.logitcrossentropy(model(x), y)
#    end
#    l/length(dataloader)
#end
#=
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end
=#



## GRU网络训练

# 模型训练参数调整
@with_kw mutable struct args
    η::Float64 = 3e-4     # 学习率
    batchsize::Int = 50   # 批量处理的数量   batch size
    epochs::Int = 10       # 训练次数
    device::Function = cpu  # set as gpu, if gpu available
end
Arg=args()


# 建立模型
m = GRU(1, 3)

# 衡量模型的结果
function eval_model(x)
    out = m.(x)[end]
    Flux.reset!(m)
    out
end
# 训练
loss(x, y) = logitcrossentropy(eval_model(x) ,y)
println("训练前损失为 = ", sum(loss.(train_input, train_out)))
opt = ADAM(0.001)

function accuracy(x, y)
    l=0
    for (i,o) in zip(x,y)
        l+=(onecold(eval_model(i))[1] == onecold(o))
    end
    l/size(x,1)
end

evalcb() = @show(sum(loss.(train_input, train_out)))
@epochs Arg.epochs Flux.train!(loss, params(m), zip(train_input,train_out), opt, cb = Flux.throttle(evalcb,1))
@show accuracy(test_input, test_out)

#= 改进: 
1. 使用词向量
2. 加一个全连接层
3. 使用时间长度不一样的训练集
=#