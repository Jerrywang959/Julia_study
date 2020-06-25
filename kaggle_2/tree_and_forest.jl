using DecisionTree, CSV, MLDataPattern, Random, DataFrames

#using Flux: onehotbatch, onecold, logitcrossentropy, throttle

# 预处理数据
function getdata()
    Trainraw_ini=CSV.read("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Train.csv")
    Testraw=CSV.read("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Test.csv")
    X=Array(Trainraw_ini[!,[:OBP,:SLG,:BA]])'
    Y=Trainraw_ini[!,:Playoffs]'

    # 分割数据集和训练集
    (X_train,y_train), (X_test,y_test) = stratifiedobs((X, Y), p = 0.9,shuffle = true,rng=MersenneTwister(42))
    
    #欠采样
    X_bal, Y_bal=MLDataPattern.oversample((X_train,y_train),fraction=0.5)
    # 过采样
    X_bal, Y_bal=MLDataPattern.undersample((X_bal,Y_bal))
    #Smote采样
    #X_bla_train, Y_bla_train = smote(hcat(Array(X_train'),Array(y_train')[:,1]), Array(y_train')[:,1], k = 2, pct_under = 100, pct_over = 200) 
    #X_bla_train=X_bla_train[:,1:3]
    # 分类编码
    #y_train, y_test = onehotbatch(Array(Y_bal)[1,:], 0:1), onehotbatch(Array(y_test)[1,:], 0:1)

    # Batching
    #train_data = DataLoader(Array(X_bal), Y_bal, batchsize=args().batchsize, shuffle=true)
    #test_data = DataLoader(Array(X_test), y_test, batchsize=args().batchsize)
    return Y_bal[:], Array(X_bal'), y_test[:], Array(X_test')
end

Y_bal, X_bal, y_test, X_test=getdata()


# 树模型
n_folds=5;
n_subfeatures=0;max_depth=10;min_samples_leaf=2;min_samples_split=5;min_purity_increase=0.0;pruning_purity=1.0

model=build_tree(Y_bal,X_bal,n_subfeatures,max_depth,min_samples_leaf,min_samples_split,pruning_purity)
model=prune_tree(model,0.8)


preds=apply_tree(model,X_test)
confusion_matrix(y_test, preds)

accuracy = nfoldCV_tree(Y_bal, X_bal,n_folds,pruning_purity,max_depth,min_samples_leaf,min_samples_split,min_purity_increase)


# 随机森林
n_subfeatures=-1;
n_trees=10;
partial_sampling=0.7;
max_depth=-1;
min_samples_leaf=5;
min_samples_split=3;
min_purity_increase=0.0
model2=build_forest(Y_bal,X_bal,n_subfeatures,n_trees,partial_sampling,max_depth,min_samples_leaf,min_samples_split,min_purity_increase)

preds2=apply_forest(model2,X_test)
confusion_matrix(y_test, preds2)

