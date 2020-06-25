# Author: 王建越 Coding:UTF-8
# 使用Julia作为主语言, 数据处理使用Julia, 模型拟合用`RCall`调用R
# Julia 环境
"""
julia> versioninfo()
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD Ryzen 5 4600U with Radeon Graphics
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, znver1)

(@v1.4) pkg> st
Status `~/.julia/environments/v1.4/Project.toml`
  [336ed68f] CSV v0.6.1
  [a93c6f00] DataFrames v0.20.2
  [6f49c342] RCall v0.13.6
"""
# R 版本   根据版本安装的对应的包
""" 
> version
               _                           
platform       x86_64-pc-linux-gnu         
arch           x86_64                      
os             linux-gnu                   
system         x86_64, linux-gnu           
status                                     
major          4                           
minor          0.2                         
year           2020                        
month          06                          
day            22                          
svn rev        78730                       
language       R                           
version.string R version 4.0.2 (2020-06-22)
nickname       Taking Off Again
"""

## julia加载包
using DataFrames,CSV,RCall
#@load LinearRegressor pkg=MLJLinearModels
#@load DecisionTreeClassifier pkg=DecisionTree
#@load DecisionTreeRegressor pkg=DecisionTree
# R加载包
R"""
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(caTools)
library(randomForest)
library(MASS)
library(ISLR)
"""

## 手动写入分区信息

NL_W=["LAD","ARI","COL","SDP","SFG"]
NL_E=["ATL","MIA","NYM","PHI","WSN"]
NL_C=["CHC","CIN","MIL","PIT","STL","HOU"]
AL_W=["CAL","OAK","SEA","TEX"]
AL_E=["BAL","BOS","NYY","TBR","TOR"]
AL_C=["CHW","CLE","DET","KCR","MIN"]

MLB=(NL_W,NL_E,NL_C,AL_W,AL_E,AL_C)
 
#=2003-2011年规则: 分区冠军+除分区冠军外胜率第一
2012年规则: 分区冠军+各联盟除分区冠军外胜率第一和第二pk, 胜者进入季后赛
因此选择预测胜场数W
=#

Trainraw_ini=CSV.read("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Train.csv")
Testraw_ini=CSV.read("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Test.csv", copycols=true)
Trainraw=dropmissing(Trainraw_ini[!,[:OBP,:SLG,:BA,:OOBP,:OSLG,:RA,:RS,:W,:Playoffs]])    #删除缺失值
Testraw=Testraw_ini[!,[:Id,:League,:Team,:BA,:Year,:OBP,:SLG,:OOBP,:OSLG,:Playoffs]]
Testraw_for_R=Testraw_ini[!,[:Id,:Year,:OBP,:SLG,:BA,:OOBP,:OSLG]]

#x1,y=Array(Trainraw[!,[:OBP,:SLG,:OOBP,:OSLG]]),Trainraw[!,:W]
#x2=Testraw[!,[:OBP,:SLG,:OOBP,:OSLG]]

## 数据导入R
@rput Trainraw
@rput Testraw_for_R

##  R线性回归
function linearreg()
    R"""
    lm.fit=lm(W ~ OBP + SLG + BA + OOBP + OSLG,data = Trainraw)
    Y=predict(lm.fit,newdata=Testraw_for_R)
    """
    @rget Y
end



## R回归树
function reg_tree()
    R"""
    fitControl = trainControl(method = "cv", number = 3)
    cpGrid = expand.grid(.cp = (1:50)*0.001)
    set.seed(33)
    cvResults = train(W ~ OBP + SLG + BA + OOBP + OSLG , data = Trainraw, method = "rpart", trControl = fitControl, tuneGrid = cpGrid)
    cvResults
    p=cvResults["bestTune"]
    TreeCV = rpart(W ~ OBP + SLG + BA + OOBP + OSLG, data = Trainraw, control = rpart.control(cp = cvResults["bestTune"]))
    prp(TreeCV)
    predictCV = predict(TreeCV, newdata = Testraw_for_R, type = "vector")
    """
    @rget p
    print(p)
    @rget predictCV
end


## R随机森林回归树
function rantree()
    R"""
    set.seed(333)
    Forest = randomForest(W ~ OBP + SLG +BA + OOBP + OSLG , data = Trainraw, ntree = 2000, nodesize = 50, mtry = 3)
    predictForest = predict(Forest, newdata = Testraw_for_R)
    """
    #取出随机森林的结果
    @rget predictForest
end


## R随机森林分类树
#R"""
#Trainraw$Playoffs=as.factor(Trainraw$Playoffs)
#set.seed(333)
#Forest = randomForest(Playoffs ~ OBP + SLG + OOBP + OSLG, data = Trainraw, ntree = 1000, nodesize = 50, mtry = 30)
#predictForest = predict(Forest, newdata = Testraw,type="prob")
#"""
#取出随机森林的结果
#@rget predictForest
#predictForest=predictForest[:,2]

##

# 分类编码
    #y_train, y_test = onehotbatch(Array(Y_bal)[1,:], 0:1), onehotbatch(Array(y_test)[1,:], 0:1)


#function train1()   #线性回归
#    linear=LinearRegressor()
#    mode1=machine(linear,Array(x1),y)
#    fit!(mode1)
#    Out=predict(mode1,x2)
#end

#function train2()  #决策树分类
#    tree_model = DecisionTreeClassifier();
#    r1=range(tree_model,:min_samples_leaf, lower=1, upper=5)
#    r2=range(tree_model,:min_samples_split, lower=2, upper=10)
#    r3=range(tree_model,:min_purity_increase, lower=0.0, upper=1.0)
###    r4=range(tree_model,:merge_purity_threshold,lower=0.0,upper=0.8)
#    r=[r1,r2,r3,r4]
#    self_tuning_tree_model = TunedModel(model=tree_model,resampling = CV(nfolds=3),tuning = Grid(resolution=10),range = r,check_measure=false);
#    self_tuning_tree= machine(self_tuning_tree_model, x1, y)
#    fit!(self_tuning_tree, verbosity=0);
#
#end    

# 不考虑联盟和分区的数据转化
function W2Playoffs()
    Testraw[!,:W].=Out        # 先把W那一列换成模型的结果
    newplay=zeros(Int64,size(Testraw)[1])   #创建一个新的向量准备替代
    Year=2003:2012           # 确定年份范围
    for i in Year              #   循环
        index=findall(x->isequal(x,i),Testraw[!,:Year])    # 找到所有为那个年的索引
        loca=index[1]       # 索引的第一个为这一场比赛的索引位置
        newdf=Testraw[index,:W]   # 调出这一年的比赛的预测结果
        sortind=sortperm(newdf,rev=true)   # 获取对newdf的排序的索引, 最大的为第一个
        newplay[sortind[1:8].+loca.-1].=1         # 最大的前八个的索引+这一场比赛的位置, 这些结果替换成1
        newplay[sortind[9:end].+loca.-1].=0       # 其他替换成0    ,  -1用来调整位置
    end
    Testraw[!,:Playoffs].=newplay         ## 先把W那一列整体替换
end

# 考虑分区等详细规则的, 先不考虑2012年规则的变化
function W2Playoffs_area()
    Wininde=[]         # 进入季后赛的队伍的索引
    Testraw[!,:W].=Out        # 先把W那一列换成模型的结果
    newplay=zeros(Int64,size(Testraw)[1])   #创建一个新的向量准备替代
    Year=2003:2012           # 确定年份范围
    for i in Year
        Playoffsindex=[]   # 预先分配一个进入季后赛的索引
        index=findall(x->isequal(x,i),Testraw[!,:Year])    # 找到所有为那个年的索引
        loca=index[1]-1   # 找到该年份比赛的相对位置
        for j in MLB  # 找每个小赛区
            areaindex=findall(x->in(x,j),Testraw[index,:Team])  #找到该年属于该分区的队伍
            (value,place)=findmax(Testraw[(areaindex.+loca),:W]) #找到最大胜场的队伍
            push!(Playoffsindex,areaindex[place]+loca)   #把该队伍的索引push进入总索引
        end
        #复制该年比赛队伍的索引并删除已经进入季后赛的索引
        achampionindex=copy(index)
        deleteat!(achampionindex,sort(Playoffsindex.-loca))
        conindex=indexin(achampionindex,index)      # 找到新的索引在旧的索引中的位置
        for r in ("NL","AL")
            wkindex=findall(x->isequal(x,r),Testraw[achampionindex,:League])   #找到属于这里联盟的队伍
            realindex=conindex[wkindex]     # 这些队伍在一年的比赛中的实际的索引
            sortinde=sortperm(Testraw[(realindex.+loca),:W],rev=true)            # 对这些队伍的成绩进行排序, rev=true 表示从大到小
            push!(Playoffsindex,realindex[sortinde[1]].+loca)                          #把该队伍的索引push进入总索引
        end
        append!(Wininde,Playoffsindex)
    end
    Testraw[!,:Playoffs].=0
    Testraw[Wininde,:Playoffs].=1
end


## 选择一个模型进行预测W
#Out=linearreg()
#Out=reg_tree()
Out=rantree()

W2Playoffs_area()

ouput=Testraw[:,[:Id,:Playoffs]]

CSV.write("ouput.csv",ouput)