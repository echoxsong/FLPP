function [predicted_train, predicted_test] = run_GbFA(train_data,test_data, train_labels,k,t, reduced_dimension)
% Input:
% train_data : numtrain * dimtrain
% test_data : numtest * dimtest
% train_labels : numtrain * 1
% reduced_dimension : the dimension that should be reduced to
% 
% Output:
% predicted_train: training data in low-dimensional space
% predicted_test : testing data in low-dimensional space
% 
% Other FUNCTIONAL parameters(settled):
% r = 10^(-3); norder = 4; nbasis = 50; Lfd = 2;
% in hyperspectral experiments, they are all settled. 
% They can be changed to other numerical numbers.

options = [];
options.t = t;%1; 
options.k = k;
options.gnd = train_labels;   
[W,W_Prime] = constructTwoW(train_data, options); % some problems here

D = diag(sum(W, 2));
L = D - W;
D_Prime = diag(sum(W_Prime, 2));
L_Prime = D_Prime - W_Prime;

[eigvector, ~] = GbFA(train_data, L, L_Prime);
eigvector = eigvector(:,1:reduced_dimension);
predicted_train = train_data * eigvector;
predicted_test = test_data * eigvector;

end