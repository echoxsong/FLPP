function [predicted_train, predicted_test] = run_LPP(train_data,test_data,k,t,reduced_dimension)
% Input:
% train_data : numtrain * dimtrain
% test_data : numtest * dimtest
% reduced_dimension : the dimension that should be reduced to
% 
% Output:
% predicted_train: training data in low-dimensional space
% predicted_test : testing data in low-dimensional space
%

options = [];
options.ReducedDim = reduced_dimension;
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';
options.t = t;
W = constructW(train_data, options);

[eigvector, ~] = LPP(W, options, train_data);
predicted_train = train_data * eigvector;
predicted_test = test_data * eigvector;

end