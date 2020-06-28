# Author: 王建越  Coding:UTF-8
# 使用Julia作为主语言, 数据处理使用Julia, 模型拟合用`RCall`调用R
#=
Idea: 胜场次数W直接决定是否进入季后赛, 因此只需要预测胜场次数W就行

2003-2011年规则: 分区冠军+除分区冠军外胜率第一
2012年规则: 分区冠军+各联盟除分区冠军外胜率第一和第二pk, 胜者进入季后赛
不考虑2012年规则的变化,规则为: 分区冠军+两个联盟除分区冠军外胜率第一

数据选择:棒球是轮回攻守,而不是实时对抗. 是否W不仅取决于自己的进攻能力,还取决于别人的进攻能力,因此只有3年的数据,不分割训练集和测试集

尝试多个模型后,还是随机森林好用
=#
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

## 修改数据的路径

Trainpath="/home/jerrywang/Desktop/kaggle_2_Train.csv"
Testpath="/home/jerrywang/Desktop/kaggle_2_Test.csv"

## julia加载包
using DataFrames,CSV,RCall
#@load LinearRegressor pkg=MLJLinearModels
#@load DecisionTreeClassifier pkg=DecisionTree
#@load DecisionTreeRegressor pkg=DecisionTree
# R加载包
R"""
library(randomForest)
"""

## 手动写入分区信息
NL_W=["LAD","ARI","COL","SDP","SFG"]
NL_E=["ATL","MIA","NYM","PHI","WSN"]
NL_C=["CHC","CIN","MIL","PIT","STL","HOU"]
AL_W=["CAL","OAK","SEA","TEX"]
AL_E=["BAL","BOS","NYY","TBR","TOR"]
AL_C=["CHW","CLE","DET","KCR","MIN"]
MLB=(NL_W,NL_E,NL_C,AL_W,AL_E,AL_C)
 
# 加载数据
Trainraw_ini=CSV.read(Trainpath)
Testraw_ini=CSV.read(Testpath, copycols=true)
Trainraw=dropmissing(Trainraw_ini[!,[:OBP,:SLG,:BA,:OOBP,:OSLG,:RA,:RS,:W,:Playoffs]])    #删除缺失值
Testraw=Testraw_ini[!,[:Id,:League,:Team,:BA,:Year,:OBP,:SLG,:OOBP,:OSLG,:Playoffs]]
Testraw_for_R=Testraw_ini[!,[:Id,:Year,:OBP,:SLG,:BA,:OOBP,:OSLG]]

## 数据导入R
@rput Trainraw
@rput Testraw_for_R

## R随机森林回归树
function rantree()
    R"""
    set.seed(333)
    Forest = randomForest(W ~ OBP + SLG +BA + OOBP + OSLG , data = Trainraw, ntree = 500, nodesize = 40, mtry = 3)
    predictForest = predict(Forest, newdata = Testraw_for_R)
    """
    #取出随机森林的结果
    @rget predictForest
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

## 随机森林预测
Out=rantree()
W2Playoffs_area()
ouput=Testraw[:,[:Id,:Playoffs]]
CSV.write("ouput.csv",ouput)