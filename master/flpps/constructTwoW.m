function [W,W_Prime] = constructTwoW(fea,options)
%	Usage:
%	W = constructW(fea,options)
%
%	fea: Rows of vectors of data points. Each row is x_i
%   options: Struct value in Matlab. The fields in options that can be set:
%                  
%           NeighborMode -  Indicates how to construct the graph. Only 'Supervised'
%               'Supervised'      -  k = 0
%                                       For W: Put an edge between two nodes if and
%                                       only if they belong to same class. 
%                                       For W_Prime: Put an edge between two nodes if and only if they do
%                                       not belong to same class.                                    
%                                    k > 0
%                                       For W: Put an edge between two nodes if they belong 
%                                       to same class and they are among the k nearst neighbors of each other. 
%                                       For W_Prime: Put an edge between two nodes if and only if they do
%                                       not belong to same class and they are among the k nearst neighbors of each other. 
%                                    Default: k=0
%                                   You are required to provide the label
%                                   information gnd in the options.
%           WeightMode   -  Indicates how to assign weights for each edge
%                           in the graph.(Only 'HeatKernel')
%               'HeatKernel'   - For W & W_Prime: If nodes i and j are connected, put weight
%                                W_ij = exp(-norm(x_i - x_j)/2t^2). You are 
%                                required to provide the parameter t. [Default One]
%            k         -   Default will be 0.
%            gnd       -   The parameter needed under 'Supervised'
%                          NeighborMode.  Colunm vector of the label
%                          information for each data point.
%            bLDA      -   0 or 1. Only effective under 'Supervised'
%                          NeighborMode. If 1, the graph will be constructed
%                          to make LPP exactly same as LDA. Default will be
%                          0. 
%            t         -   The parameter needed under 'HeatKernel'
%                          WeightMode. Default will be 1
%      bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 0
%                          if 'Supervised' NeighborMode & bLDA == 1,
%                          bSelfConnected will always be 1. Default 0.
%
%       He, Xiaofei, and Partha Niyogi. "Locality preserving projections." Advances in neural information processing systems. 2004.
% 
%       Put an edge between two points if they do not belong to the same
%       class. Using heat kernal function. (Penalty graph)
%       Cui Y, Fan L. "A novel supervised dimensionality reduction algorithm: Graph-based Fisher analysis[J]". Pattern Recognition, 2012, 45(4): 1471-1481.
%       
%       Modified by Xin Song on 2/5/2018 & 30/5/2018

if (~exist('options','var'))
   options = [];
end
%=================================================
% if ~isfield(options,'NeighborMode')
%     options.NeighborMode = 'Supervised';
% end

if ~isfield(options,'k')
    options.k = 0;
end

% if ~isfield(options,'bLDA')
%     options.bLDA = 0;
% end

% if options.bLDA
%     options.bSelfConnected = 1;
% end

if ~isfield(options,'gnd')
    error('Label(gnd) should be provided under supervised learning!');
end

if ~isempty(fea) && length(options.gnd) ~= size(fea,1)
    error('gnd doesn''t match with fea!');
end

% if ~isfield(options,'WeightMode')
%     options.WeightMode = 'HeatKernel';
% end

if ~isfield(options,'t')
    nSmp = size(fea,1);
    if nSmp > 3000
        D_Prime = EuDist2(fea(randsample(nSmp,3000),:));
    else
         D_Prime = EuDist2(fea);
    end
    options.t = mean(mean(D_Prime));
end

if ~isfield(options,'bSelfConnected')
    options.bSelfConnected = 0;
end


%=================================================
nSmp = length(options.gnd);
Label = unique(options.gnd);
nLabel = length(Label);
% if options.bLDA
%     G = zeros(nSmp,nSmp);
%     for idx=1:nLabel
%         classIdx = options.gnd==Label(idx);
%         G(classIdx,classIdx) = 1/sum(classIdx);
%     end
%     W = sparse(G);
%     return;
% end
if options.k > 0
    G = zeros(nSmp*(options.k+1),3);
    G_Prime = zeros(nSmp*(options.k),3);
    idNow = 0;
    idxNow_Prime = 0;
    for i=1:nLabel
        classIdx = find(options.gnd==Label(i));
        nLengthClass = length(classIdx);
        if nLengthClass <= options.k
            % should notice the number of samples in Label(i)
            % it may be smaller than k
            D = EuDist2(fea(classIdx,:),[],0);
            [dump,idx] = sort(D,2); % sort each row
            clear D
            
            dump = exp(-dump/(2*options.t^2));
            nSmpClass = nLengthClass *(options.k+1);  
            G(idNow+1:nLengthClass*nLengthClass+idNow,1) = repmat(classIdx, [nLengthClass,1]);
            G(idNow+1:nLengthClass*nLengthClass+idNow,2) = classIdx(idx(:));
            G(idNow+1:nLengthClass*nLengthClass+idNow,3) = dump(:);
            idNow = idNow+nSmpClass;
            clear dump idx
        else
            D = EuDist2(fea(classIdx,:),[],0); %W
            [dump,idx] = sort(D,2); % sort each row
            clear D     
            
            idx = idx(:,1:options.k+1);
            dump = dump(:,1:options.k+1);
            dump = exp(-dump/(2*options.t^2));
        
            nSmpClass = length(classIdx)*(options.k+1);  
            G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
            G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
            G(idNow+1:nSmpClass+idNow,3) = dump(:);
            idNow = idNow+nSmpClass;
            clear dump idx
        end
    end
    for i = 1:nLabel
        classIdx = find(options.gnd==Label(i));
        classIdx_No = find(options.gnd~=Label(i));
        D_Prime = EuDist2(fea(classIdx,:),fea(classIdx_No,:),0);  % W_Prime
        [dump_Prime,idx_Prime] = sort(D_Prime,2); % sort each row
        clear D_Prime
        
        idx_Prime = idx_Prime(:,1:options.k);
        dump_Prime = dump_Prime(:,1:options.k);
        dump_Prime = exp(-dump_Prime/(2*options.t^2));
        nSmpClass_Prime = length(classIdx)*(options.k);
        G_Prime(idxNow_Prime+1:nSmpClass_Prime+idxNow_Prime,1) = repmat(classIdx,[options.k,1]);
        G_Prime(idxNow_Prime+1:nSmpClass_Prime+idxNow_Prime,2) = classIdx_No(idx_Prime(:));
        G_Prime(idxNow_Prime+1:nSmpClass_Prime+idxNow_Prime,3) = dump_Prime(:);
        idxNow_Prime = idxNow_Prime+nSmpClass_Prime;
        clear dump_Prime idx_Prime
    end
    idx_Zero = G(:,1) == 0; %delete the blank row
    G(idx_Zero,:) = [];
    G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    G_Prime = sparse(G_Prime(:,1),G_Prime(:,2),G_Prime(:,3),nSmp,nSmp);
else
    G = zeros(nSmp,nSmp);
    G_Prime = zeros(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(options.gnd==Label(i));
        classIdx_No = find(options.gnd~=Label(i));
        D = EuDist2(fea(classIdx,:),[],0);
        D = exp(-D/(2*options.t^2));
        G(classIdx,classIdx) = D;
        D_Prime = EuDist2(fea(classIdx,:),fea(classIdx_No,:),0);
        D_Prime = exp(-D_Prime/(2*options.t^2));
        G_Prime(classIdx,classIdx_No) = D_Prime;
    end
end

if ~options.bSelfConnected
    for i=1:size(G,1)
        G(i,i) = 0;
    end
%     for i=1:size(G_Prime,1)
%         G_Prime(i,i) = 0;
%     end
end

W = sparse(max(G,G'));
W_Prime = sparse(max(G_Prime,G_Prime'));

end