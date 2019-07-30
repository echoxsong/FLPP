function [predicted_train, predicted_test] = run_SFLPP(train_data,test_data,train_labels,kk,t,reduced_dimension)
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

[~,dimtrain] = size(train_data);

r = 10^(-3); 
rangewavelength = r * [1, dimtrain]';
norder = 5;% b-spline order
wavelength = r * (1:dimtrain)';   % wavelength
nbasis = 50;
Lfd = 2;

% create bspline basis
basisobj = create_bspline_basis(rangewavelength, nbasis, norder);
%     fdnames{1} = 'Wavelength';
%     fdnames{2} = 'What 1';
%     fdnames{3} = 'What 2';
% choose lambda
lnlam = -8:1:0;
gcvsave = zeros(length(lnlam),1);

for i=1:length(lnlam)
    fdParobj = fdPar(basisobj, Lfd, 10^lnlam(i));
    [~, ~, gcv] = smooth_basis(wavelength, train_data', fdParobj, []);
    gcvsave(i) = sum(gcv);
end
[~, k] = max(-gcvsave);
lambda = 10^lnlam(k);  % best lambda

% roughness penalty method
fdParobj = fdPar(basisobj, Lfd, lambda);
[fdobj_train, ~, ~] = smooth_basis(wavelength, train_data', fdParobj, []);%;, fdnames);
[fdobj_test, ~, ~] = smooth_basis(wavelength, test_data', fdParobj, []);%;, fdnames);

coef = getcoef(fdobj_train);
options = [];
options.t = t;%filter:1    nofilter:0.5;
options.gnd = train_labels;
options.k = kk; % paviaU 3 best
[W,W_Prime] = constructTwoW(coef', options);

D = diag(sum(W, 2));
L = D - W;
D_Prime = diag(sum(W_Prime, 2));
L_Prime = D_Prime - W_Prime;

%     max_nharm = 36;
flppstr_train = lpp_fd_Supervised(fdobj_train, L, L_Prime, reduced_dimension, fdParobj, 0);
predicted_train = flppstr_train.harmscr;
predicted_test = inprod(fdobj_test, flppstr_train.harmfd);

end