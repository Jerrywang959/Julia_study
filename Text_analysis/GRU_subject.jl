## 数据来源:  https://www.datafountain.cn/competitions/310/datasets
## 词向量来源: https://github.com/Embedding/Chinese-Word-Vectors

using CSV,DataFrames,PyCall,Statistics,JLD,Lathe
using Lathe.preprocess: TrainTestSplit
using Flux
using Flux: onehotbatch,logitcrossentropy,onecold,@epochs,RNN,GRU,chunk,reset!
using Flux.Data: DataLoader
using Parameters: @with_kw


## 导入python模型TLP
ltp=pyimport("ltp").LTP()

## 转化python模型为函数,方便输出,其中输入为字符串,
function getword(x::String)::Array{String,1}
    seg,_=ltp.seg([x])
    return seg[:]
end

## 读取数据
Trainraw=CSV.read("/home/jerrywang/Desktop/car_data/Final_project.csv")
aim=Trainraw[!,:sentiment_value]
aim2=Trainraw[!,:subject]


### 主题词典
Subjetct=Dict()
Subjetct["安全性"]=1
Subjetct["操控"]=2
Subjetct["动力"]=3
Subjetct["价格"]=4
Subjetct["空间"]=5
Subjetct["内饰"]=6
Subjetct["配置"]=7
Subjetct["价格"]=8
Subjetct["舒适性"]=9
Subjetct["外观"]=10
Subjetct["油耗"]=11
tran_aim2=map(x->Subjetct[x],aim2)
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
model=readlines("/home/jerrywang/Desktop/car_data/sgns.weibo.bigram-char")[2:end]
modelDict=Dict{String,Array{Float64,1}}()
# 注: 此步骤并行存在问题,还是多用矩阵运算比较好
for i in model
    u=split(i," ")
    modelDict[u[1]]=parse.(Float64,u[2:end-1])
end

# 一个词语转化为向量  不在语料库里则转为0
word2vec(x::String)::Array{Float64,1}=x in keys(modelDict) ? modelDict[x] : zeros(Float64,300)
# 一个句向量转化为一系列词向量
sentence2vec(x::Array{String,1})::Array{Array{Float64,1},1}=map(x->word2vec.(x),x)
# 完成转换
WordVector=by(PureWord_lengthed,:content_id, vec=:wordlengthed=>x->sentence2vec.(x))
WordVector2=copy(WordVector)
WordVector.y=aim

WordVector2.yy=tran_aim2


## 分割数据集

train,test= Lathe.preprocess.TrainTestSplit(WordVector, 0.75)
train2,test2= Lathe.preprocess.TrainTestSplit(WordVector2, 0.75)
#暂存结果,方便下次使用
@save "/home/jerrywang/Desktop/var.jld" train test
@load "/home/jerrywang/Desktop/var.jld"

@save "/home/jerrywang/Desktop/var2.jld" train2 test2
@load "/home/jerrywang/Desktop/var2.jld"

