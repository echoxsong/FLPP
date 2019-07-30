function flppstr = lpp_fd_Gao(fdobj, L, D, nharm, harmfdPar, centerfns)
%  FLPP Functional locality preserving projection
%
%  Arguments:
%  FDOBJ     ... Functional data object (a struct object)
%  NHARM     ... Number of principal components to be kept. Default 2
%  HARMFDPAR ... A functional parameter object specifying the
%                basis, differential operator, and level of smoothing
%                for eigenfunctions.
%  L         ... Laplacian matrix L = D - S.
%  CENTERFNS ... If 1, the mean function is first subtracted from each 
%                function.  1 is the default. ?
%
%  Returns:
%  A struct object FLPPSTR with the fields:
%  HARMFD  ... A functional data object for the harmonics or eigenfunctions
%  VALUES  ... The complete set of eigenvalues
%  HARMSCR ... A matrix of scores on the principal components or harmonics
%  VARPROP ... A vector giving the proportion of variance explained
%                 by each eigenfunction
%  FDHATFD ... A functional data object for the approximation to the
%              FDOBJ based on NHARM principal components
%  MEANFD  ... A functional data object giving the mean function
%
%  If NHARM = 0, all fields except MEANFD are empty.
%----------------------------%%%%%--------------------------------%

%  check FDOBJ
if ~isa_fd(fdobj)
    error ('First argument is not a functional data object.');
end

%  get basis information for functional data
fdbasis  = getbasis(fdobj);

%  set up default values   % Why??? Junbin
if nargin < 5
    centerfns = 1;   %  subtract mean from data before FLPP
end

if nargin < 4
    %  default Lfd object: penalize 2nd deriv., lambda = 0
    Lfdobj    = int2Lfd(2);
    lambda    = 0;
    harmfdPar = fdPar(fdbasis, Lfdobj, lambda);
else
    %  check harmfdPar object
    if ~isa_fdPar(harmfdPar)
        if isa_fd(harmfdPar) || isa_basis(harmfdPar)
            harmfdPar = fdPar(harmfdPar);
        else
            error(['HARMFDPAR is not a functional parameter object, ', ...
                'not a functional data object, and ', ...
                'not a basis object.']);
        end
    end
end

if nargin < 3
    nharm = 2;  %  default to two harmonics
end

%  compute mean function
%  Why?   Junbin
meanfd = mean(fdobj);

if nharm == 0
    flppstr.harmfd  = [];
    flppstr.values  = [];
    flppstr.harmscr = [];
    flppstr.varprop = [];
    flppstr.fdhatfd = [];
    flppstr.meanfd  = meanfd;
    return
end

%  ---------   begin functional locality preserving projection  ----------

% center data if required
if centerfns ~= 0
    fdobj = center(fdobj);
end

%  set up HARMBASIS
harmbasis = getbasis(getfd(harmfdPar));
% nhbasis   = getnbasis(harmbasis);

%  set up LFDOBJ
% Lfdobj = getLfd(harmfdPar);
% Lfdobj = int2Lfd(Lfdobj); %if not lfdobj, then transform; if yes, then return

%  get coefficient matrix for FDOBJ and its dimensions
coef   = getcoef(fdobj);
coefd  = size(coef);
nbasis = coefd(1);
nrep   = coefd(2);
ndim   = length(coefd);

if nrep < 2
    error('FLPP not possible without replications');
end

%  compute CTEMP whose cross product is needed
%  This is suspicious
if ndim == 3
    nvar  = coefd(3);
    Cmat = zeros(nvar*nbasis,nrep);
    for j = 1:nvar
        index = (1:nbasis) + (j-1)*nbasis;
        Cmat(index,:) = coef(:,:,j);
    end
else
    nvar = 1;
    Cmat = coef; 
end

%  compute inner product of eigenfunction basis and functional
%  data basis. W = \int_{T}\phi_k(t)\phi_l{t} dt
W = inprod_basis(harmbasis, fdbasis);
W = 0.5*(W + W');   % Make sure it is symmetric

%  set up matrix for eigenanalysis
%  ?consider multivariable?
%  method 1: generalized eigenvalue problem
if nvar == 1
    Lmat = W * Cmat * L * Cmat' * W;
    %Rmat = W;
    Rmat = W * Cmat * D * Cmat' * W;
    Lmat = (Lmat + Lmat') / 2;
    Rmat = (Rmat + Rmat') / 2;
    [eigvector, eigvalue] = eig(Lmat, Rmat);
else  % The following construction is not correct!  Junbin Gao
    Lmat = zeros(nvar*nbasis,nbasis);
    Rmat = zeros(nvar*nbasis,nbasis);
    eigvector = zeros(nbasis,nbasis,nvar);
    eigvalue = zeros(nbasis,nbasis,nvar);
    for i = 1:nvar
        indexi =   (1:nbasis) + (i-1)*nbasis;
        Lmat(indexi,:) = W * Cmat(indexi,:) * L * Cmat(indexi,:)' * W;
        Rmat(indexi,:) = W;
        [eigvecs, eigvals] = eig(Lmat(indexi,:), Rmat(indexi,:));
        eigvector(:,:,i) = eigvecs;
        eigvalue(:,:,i) = eigvals;
    end
end

%  method 2: transform from generalized eigenvalue problem to normal 
%  ?consider multivariable?
%  eigenvalue problem. a = W^-1/2*b . Since W is positive semi-definite.
%  ?complex data?
% % if nvar == 1
% %     Lmat = W^(1/2) * Cmat * L * Cmat' * W^(1/2);
% %     [eigvecs, eigvalue] =eig(Lmat);
% %     eigvector = W^(-1/2) * eigvecs; %eigenvector: column vector
% % else
% %     Lmat = zeros(nvar*nbasis,nbasis);
% %     eigvector = zeros(nbasis,nbasis,nvar);
% %     eigvalue = zeros(nbasis,nbasis,nvar);
% %     for i = 1:nvar
% %         indexi =   (1:nbasis) + (i-1)*nbasis;
% %         Lmat(indexi,:) = W^(1/2) * Cmat(indexi,:) * L * Cmat(indexi,:)' * W^(1/2);
% %         [eigvecs, eigvals] = eig(Lmat(indexi,:));
% %         eigvalue(:,:,i) = eigvals;
% %         eigvector(:,:,i) = W^(-1/2) * eigvecs;
% %     end
% % end

%  select nharm eigenvector and eigenvalue
if nvar == 1
    [eigvalue, indsrt ] = sort(diag(eigvalue));  %,'descend');
    eigvector = eigvector(:,indsrt);
    indx = (1:nharm);
    eigvector = eigvector(:,indx);
    varprop = eigvalue(1:nharm)./sum(eigvalue);
else
    varprop = zeros(nharm,nvar);
    for i = 1:nvar
        [eigvalue(:,i), indsrt ] = sort(diag(eigvalue(:,:,i)));
        eigvector(:,:,i) = eigvector(:,indsrt,i);
        indx = (1 : nharm);
        eigvectemp(:,:,i) = eigvector(:,indx,i); %#ok<AGROW>
        varprop(:,i) = eigvalue(1:nharm,i)./sum(eigvalue(:,i));
    end
    eigvector = eigvectemp;
end

%  Set up fdnames for harmfd
harmnames = getnames(fdobj);
%  Name and labels for harmonics
harmlabels = ['I   '; 'II  '; 'III '; 'IV  '; 'V   '; ...
              'VI  '; 'VII '; 'VIII'; 'IX  '; 'X   '];
if nharm <= 10
    harmnames2    = cell(1,2);
    harmnames2{1} = 'Harmonics';
    harmnames2{2} = harmlabels(1:nharm,:);
    harmnames{2}  = harmnames2;
else
    harmnames{2} = 'Harmonics';
end
%  Name and labels for variables
if nvar ~= 1
    if iscell(harmnames{3})
        harmnames3    = harmnames{3};
        harmnames3{1} = ['Harmonics for ',harmnames3{1}];
        harmnames{3}  = harmnames3;
    else
        if ischar(harmnames{3}) && size(harmnames{3},1) == 1
            harmnames{3} = ['Harmonics for ',harmnames{3}];
        else
            harmnames{3} = 'Harmonics';
        end
    end
end

%  set up harmfd. harmcoef == eigenvector.
harmfd = fd(eigvector, harmbasis, harmnames);

%  set up harmscr
if nvar == 1
    harmscr = inprod(fdobj, harmfd);
else
    harmscr = zeros(nrep, nharm, nvar);
%     coef   = getcoef(fdobj);
    for j = 1:nvar
        coefj = squeeze(coef(:,:,j));%Remove singleton dimensions
        harmcoefj = squeeze(eigvector(:,:,j));
        fdobjj = fd(coefj,fdbasis);
        harmfdj = fd(harmcoefj,fdbasis);
        harmscr(:,:,j) = inprod(fdobjj,harmfdj);
    end
end

% set up structure object flppstr
flppstr.harmfd = harmfd;
flppstr.values = eigvalue;
flppstr.harmscr = harmscr;
flppstr.varprop = varprop;
flppstr.meanfd = meanfd;
end