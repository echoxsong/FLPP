%% plotSVMroc_test

%%
clear;
clc;
%%
load wine_test

%%
[train_data,test_data] = scaleForSVM(train_data,test_data,0,1); 
model = svmtrain(train_data_labels,train_data);

[pre,acc] = svmpredict(test_data_labels,test_data,model);

%% plotSVMroc
plotSVMroc(test_data_labels,pre,3)
