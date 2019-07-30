function [OverallAccuracy, AverageAccuracy, Kappa, PerAcc, Allpredict_label] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, classifierType)
% Input:
% predicted_train: numtrain * reduced_dimension
% predicted_test: numtest * reduced_dimension
% train_labels: numtrain * 1
% test_labels: numtest * 1
% classifierType: There are two classifiers, KNN(default) and SVM. 
% Output:
% OverallAccuracy: Overall accuracy
% AverageAccuracy: Average accuracy
% Kappa: kappa-Cohen's kappa

% set up default values  
if nargin < 5
    classifierType = KNN;
end

% Initialization 
[numtest, reduced_dimension] = size(predicted_test);
nclass = length(unique(train_labels));
OverallAccuracy = zeros(1,30);
AverageAccuracy = zeros(1,30);
Kappa = zeros(1,30);
PerAcc = zeros(reduced_dimension,nclass);
Allpredict_label = zeros(numtest, 30);

% ------------begin calculating ---------------------
for iDimension = 1 : reduced_dimension
    % build classifier model
    train_final = predicted_train(:,1:iDimension);
    test_final = predicted_test(:,1:iDimension);
    switch classifierType
        case 'KNN'
            Mdl = fitcknn(train_final,train_labels,'NumNeighbors',5);
            predict_label = predict(Mdl,test_final);
        case 'SVM'
            % method 1:
            c_vec = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000;];
            g_vec = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000;];
%             c_vec = [10000;];
%             g_vec = [1;];
            bestcg = zeros(1,2);
%             [train_final,test_final] = scaleForSVM(train_final,test_final,0,1);
            accuracyTemp = 0;
            for i = 1:8
                for j = 1:8
                    cmd = ['-q -t 2 -s 0 -b 1 -c ',num2str(c_vec(i)),' -g ',num2str(g_vec(j))];
                    model = svmtrain(train_labels, train_final, cmd); %#ok<SVMTRAIN>
                    [predict_label_temp, accuracy, ~] = svmpredict(test_labels, test_final, model, '-b 1');
                    if accuracy(1,1) > accuracyTemp
                        accuracyTemp = accuracy(1,1);
                        predict_label = predict_label_temp;
                        bestcg(1,1) = c_vec(i);
                        bestcg(1,2) = g_vec(j);
                    end
                end
            end
            % method 2:
%             [train_final,test_final] = scaleForSVM(train_final,test_final,0,1);
%             [bestc, bestg, ~] = automaticParameterSelection2(train_labels, train_final, 5);
% %             [bestc, bestg, ~] = SVMcgForClass(train_labels, train_final,-4,8,-4,4,3,0.5,0.5);
% %             [bestc, bestg, ~] = SVMcgForClass(train_labels, train_final,0,8,0,4,5,0.5,0.5);
%             if bestc == 0
%                 bestc = 10000;
%             end
% 
%             cmd = ['-q -t 2 -s 0 -b 1 -c ',num2str(bestc),' -g ',num2str(bestg)];
%             model = svmtrain(train_labels, train_final, cmd); %#ok<SVMTRAIN>
%             [predict_label, ~, ~] = svmpredict(test_labels, test_final, model, '-b 1');
    end
    Allpredict_label(:,iDimension) = predict_label;
    
    % Make confusion matrix for the overall classification
    [confusionMatrixAll,~] = confusionmat(test_labels,predict_label);
    TP=zeros(1,nclass);
    FN=zeros(1,nclass);
    FP=zeros(1,nclass);
    TN=zeros(1,nclass);
    for i=1:nclass
        TP(i)=confusionMatrixAll(i,i);
        FN(i)=sum(confusionMatrixAll(i,:))-confusionMatrixAll(i,i);
        FP(i)=sum(confusionMatrixAll(:,i))-confusionMatrixAll(i,i);
        TN(i)=sum(confusionMatrixAll(:))-TP(i)-FP(i)-FN(i);
    end
%     P=TP+FN;
%     N=FP+TN;
    OA = trace(confusionMatrixAll)/sum(confusionMatrixAll(:));
    AA = TP./(TP+FP);
    PerAcc(iDimension,:) = AA; % precision of each class
    AA(isnan(AA)) = 0;
    AA = mean(AA);
    
    po = OA;
    pe = sum((TP+FP) .* (TP+FN)) / (numtest*numtest);
    kappa = (po-pe)/(1-pe);
    
    OverallAccuracy(1,iDimension) = OA;
    AverageAccuracy(1,iDimension) = AA;
    Kappa(1,iDimension) = kappa;
end

ind = find(OverallAccuracy == 0);
if ~isempty(ind) 
    OverallAccuracy(1,ind) = OverallAccuracy(1,ind(1)-1);
end
ind = find(AverageAccuracy == 0);
if ~isempty(ind) 
    AverageAccuracy(1,ind) = AverageAccuracy(1,ind(1)-1);
end
ind = find(Kappa == 0);
if ~isempty(ind) 
    Kappa(1,ind) = Kappa(1,ind(1)-1);
end

end