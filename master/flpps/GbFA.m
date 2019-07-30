function [eigvector, eigvalue] = GbFA(X, L, L_Prime)

%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of LPP eigen-problem. 

%     if size(X, 2) > size(X, 1)
%         error('Number of samples should be higher than number of dimensions.');
%     end
    
    % Get P
    Fmat = X'*L*X;
    Fmat = (Fmat + Fmat') / 2;
    [Peigvector, Peigvalue] = eig(Fmat);
    Peigvector = Peigvector(:, diag(Peigvalue)>0);
    Sp = Peigvector' * X' * L_Prime * X * Peigvector;
    Sc = Peigvector' * X' * L * X * Peigvector;
%     Sp = (Sp + Sp') / 2;% useless
%     Sc = (Sc + Sc') / 2;
    [eigvector, eigvalue] = eig(Sp, Sc);
    eigvector = Peigvector * eigvector;    
%     [eigvalue, indsrt] = sort(diag(eigvalue));%,'descend');
%     eigvector = eigvector(:,indsrt);
    
    

%       LD = X' * L_Prime * X;
%       RD = X' * L * X;
%       LD = (LD + LD')/2;
%       RD = (RD + RD')/2;
%       [eigvector,eigvalue] = eig(LD, RD);
      
    
end