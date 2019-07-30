function [predicted_train, predicted_test] = run_FPCA(train_data,test_data, reduced_dimension)
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
norder = 4;% b-spline order
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
    [~, ~, gcv] = smooth_basis(wavelength, train_data', fdParobj, []);%;, fdnames);
    gcvsave(i) = sum(gcv);
end
[~, k] = max(-gcvsave);
lambda = 10^lnlam(k);  % best lambda

% roughness penalty method
fdParobj = fdPar(basisobj, Lfd, lambda);
[fdobj_train, ~, ~] = smooth_basis(wavelength, train_data', fdParobj, []);%;, fdnames);
[fdobj_test, ~, ~] = smooth_basis(wavelength, test_data', fdParobj, []);%;, fdnames);

fpcastr_train = pca_fd(fdobj_train, reduced_dimension);
predicted_train = fpcastr_train.harmscr;
predicted_test = inprod(fdobj_test - fpcastr_train.meanfd, fpcastr_train.harmfd);

end