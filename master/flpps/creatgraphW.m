function W = creatgraphW(Datatrain,train_labels,index_training,rho1,rho2,c,method)
% modified from Wei Li's code
% 8/2/2019
[N,~] = size(Datatrain); % Each row is a data point
Ctrain = tabulate(train_labels);
Ctrain = Ctrain(:,2); % number of data points in every class

W = zeros(N);
switch method
    case 'CGDA'
        a = 0;
        CMat = [];
        for i = 1 : length(Ctrain)
            Data_temp = Datatrain((a+1):(Ctrain(i)+a),:);
            a = Ctrain(i) + a;
            [NinCls,~] = size(Data_temp);
            Affinity = [];
            for j = 1 : NinCls
                Data = Data_temp;  % Data: n x d
                Y = Data(j,:);   % Y: 1 x d
                Data(j,:) = [];
                norms1 = sum((Data - repmat(Y, [size(Data,1) 1])).^2 , 2); 
                G1 = diag(rho1.*norms1);
                A = (Data * Data' + G1)\(Data*Y');
                B = zeros(1, NinCls);
                B([1:j-1, j+1:NinCls]) = A; % similarities except itself
                Affinity = [Affinity; B]; 
            end
            Cmat_each = (Affinity + Affinity')/2;
            CMat=blkdiag(CMat,Cmat_each);
        end
        W = CMat;
    case 'SaCGDA'
        a = 0;
        CMat = [];
        for i = 1 : length(Ctrain)
            Data_temp = Datatrain((a+1):(Ctrain(i)+a),:);
            index_training_temp = index_training((a+1):(Ctrain(i)+a),:);
            a = Ctrain(i) + a;
            [NinCls,~] = size(Data_temp);
            Affinity = [];
            for j = 1 : NinCls
                % spectral information
                Data = Data_temp;  % Data: n x d
                Y = Data(j,:);   % Y: 1 x d
                Data(j,:) = [];
                norms1 = sum((Data - repmat(Y, [size(Data,1) 1])).^2 , 2); 
                G1 = diag(rho1.*norms1);
                
                % spatial information
                XY = index_training_temp;
                xy = index_training_temp(j,:);
                XY(j,:) = [];
                norms2 = sum((XY - repmat(xy, [size(XY,1) 1])).^c , 2); 
                norms2 = norms2./max(norms2);
                D = diag(rho2.*norms2);
                
                A = (Data * Data' + G1 + D)\(Data*Y');
                B = zeros(1, NinCls);
                B([1:j-1, j+1:NinCls]) = A; % similarities except itself
                Affinity = [Affinity; B]; 
            end
            Cmat_each = (Affinity + Affinity')/2;
            CMat=blkdiag(CMat,Cmat_each);
        end
        W = CMat;
end

end