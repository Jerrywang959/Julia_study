using DataFrames
using CSV
using XLSX
using Dates
## 第一部分，去掉脱贫的县
B=DataFrame(XLSX.readdata("nCov_comfirmed_with_poor_Country\\poor_county.xlsx","Sheet1!A2:A852"))
tuopin=findall(x->occursin("*",x),B[!,1])
C=deleterows!(B,tuopin)    #脱贫的县不考虑
#  XLSX.writetable("pink.xlsx",REPORT_A=(collect(DataFrames.eachcol(C)), DataFrames.names(C)))
## 第二部分：贫困区县划到地级市
NE=DataFrame(Name=String[],directly_under=String[],确诊=Int64[],治愈=Int64[],死亡=Int64[],日期=Dates.Date[])
Covid=DataFrame(CSV.read("out_2.22.csv"))
quhua=DataFrame(XLSX.readdata("nCov_comfirmed_with_poor_Country\\行政区划.xlsx","Sheet1!B4:C3216"))
province_name=["安徽","甘肃","广西","贵州",
"海南","河北","河南","黑龙江","湖北","湖南",
"吉林","江西","内蒙古","宁夏","青海","山西",
"陕西","四川","西藏","新疆","云南","重庆"]
set=Set()
#去掉字符串的空格
quhua[!,2]=map(x->strip(x),quhua[!,2])

for i in 1:419
    name=B[i,1]
    #print(i)
    #print(name)
    if name in province_name
        continue
    end
    i_inde=findall(x->isequal(name,x),quhua[!,:x2])
    if length(i_inde)==0  #相等的找不到,选择包含
        i_inde=findall(x->occursin(name,x),quhua[!,:x2])
    end
    if length(i_inde)==0 #包含也找不到，选择前两个相等
        i_inde=findall(x->isequal(string(name[1],name[4]),string(x[1],x[4])),quhua[!,:x2])
    end
    if length(i_inde)==0 #前两个相等也找不到，选择前两个包含
        i_inde=findall(x->occursin(string(name[1],name[4]),x),quhua[!,:x2])
    end
    #i_inde=findall(x->length(intersect(name,x))>1,quhua[!,:x2])
    if length(i_inde) >1
        println("被选择的name=$(name)")
        linsi=DataFrame([collect(1:length(i_inde)),quhua[i_inde,:x2]])
        print(linsi)
        a=parse(Int64,input("你要选择的第几个"))
        i_inde=i_inde[end]
    end
    i_inde=i_inde[1]
    number=quhua[i_inde,:x1]
    number=number[1]
    #若区划编码为100的整数，则为地级市
    while rem(number,100)!=0
        i_inde-=1
        number=quhua[i_inde,:x1]
        number=number[1]
    end
    city=quhua[i_inde,:x2]
    #找到该地级市的情况
    print(quhua[i_inde,:x2])
    qkind=findlast2(x->isequal(x,string(city[1],city[4])),Covid[!,:市])
    print(qkind)
    push!(set,qkind)
    if qkind==nothing
        push!(NE,[name,city,0,0,0,Dates.Date(2000-1-1)])
    else
        push!(NE,[name,city,Covid[qkind,11],Covid[qkind,12],Covid[qkind,13],Covid[qkind,14]])
    end
    print(i)

end


XLSX.writetable("nCov_comfirmed_with_poor_Country\\myjieguo1.xlsx",Sheet1=(collect(DataFrames.eachcol(NE)),DataFrames.names(NE)))

##
#A=DataFrame(CSV.read("nCov_comfirmed_with_poor_Country\\DXYArea.csv"))
#describe(A)
