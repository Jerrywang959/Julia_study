using ScikitLearn,DecisionTree, CSV, MLDataPattern, Random, DataFrames
using ScikitLearn.GridSearch: GridSearchCV
using Plots, PlotlyJS
plotlyjs()
# 读取数据
Trainraw_ini=CSV.read("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Train.csv")
#可视化分析
plot(Trainraw_ini[!,[:OBG,:Playoffs]])





Trainraw=dropmissing(Trainraw_ini[!,[:OBP,:SLG,:OOBP,:OSLG,:Playoffs]])
X=Array(Trainraw[!,[:OBP,:SLG,:OOBP,:OSLG]])'
Y=Trainraw[!,:Playoffs]'

# 分割数据集和训练集
(X_train,y_train), (X_test,y_test) = stratifiedobs((X, Y), p = 0.6,shuffle = true,rng=MersenneTwister(42))

#欠采样
#X_bal, Y_bal=MLDataPattern.oversample((X_train,y_train),fraction=0.4)
# 过采样
#X_bal, Y_bal=MLDataPattern.undersample((X_bal,Y_bal))
# 不采样
X_bal, Y_bal=X_train,y_train
#Smote采样
#X_bla_train, Y_bla_train = smote(hcat(Array(X_train'),Array(y_train')[:,1]), Array(y_train')[:,1], k = 2, pct_under = 100, pct_over = 200) 
#X_bla_train=X_bla_train[:,1:3]
# 分类编码
#y_train, y_test = onehotbatch(Array(Y_bal)[1,:], 0:1), onehotbatch(Array(y_test)[1,:], 0:1)
# Batching
#train_data = DataLoader(Array(X_bal), Y_bal, batchsize=args().batchsize, shuffle=true)
#test_data = DataLoader(Array(X_test), y_test, batchsize=args().batchsize)




params=Dict(:pruning_purity_threshold => 0.2:0.1:0.9,:max_depth=>1:1:20,:min_samples_leaf=>2:1:5,:min_samples_split=>2:10)
gridsearch = GridSearchCV(DecisionTreeClassifier(), params)
ScikitLearn.fit!(gridsearch,X_bal, Y_bal)
println("Best hyper-parameters: $(gridsearch.best_params_)")
preds=predict(gridsearch,X_test);
confusion_matrix(y_test, preds)