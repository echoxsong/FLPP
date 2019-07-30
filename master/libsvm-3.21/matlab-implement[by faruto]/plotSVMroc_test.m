%% plotSVMroc_test

%%
clear;
clc;
%%
load wine_test

%%
[train_data,test_data] = scaleForSVM(train_data,test_data,0,1); 
model = svmtrain(train_data_labels,train_data,'-b 1');

[pre,acc,dec] = svmpredict(test_data_labels,test_data,model);

%% plotSVMroc
plotSVMroc(test_data_labels,dec,3)
