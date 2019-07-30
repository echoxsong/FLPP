clear;
addpath(genpath('./fdaM'))
addpath(genpath('./flpps'))
addpath(genpath('./hyperspectral data'))
addpath(genpath('./libsvm-3.21'))
%%  Indian KNN
PCA_OA = zeros(10,30); PCA_AA = zeros(10,30); PCA_Kappa = zeros(10,30);
LPP_OA = zeros(10,30); LPP_AA = zeros(10,30); LPP_Kappa = zeros(10,30);
GbFA_OA = zeros(10,30); GbFA_AA = zeros(10,30); GbFA_Kappa = zeros(10,30);
FPCA_OA = zeros(10,30); FPCA_AA = zeros(10,30); FPCA_Kappa = zeros(10,30);
FLPP_OA = zeros(10,30); FLPP_AA = zeros(10,30); FLPP_Kappa = zeros(10,30);
SFLPP_OA = zeros(10,30); SFLPP_AA = zeros(10,30); SFLPP_Kappa = zeros(10,30);
for i = 1 : 10
    [train_data,train_labels,test_data,test_labels,~, ~, ~, ~, ~, ~] = ChooseRSdata('Indianpines', 0.05, i, 1, 7);
    %========================PCA================================   
    options.ReducedDim = 30;
    [eigvector,~] = PCA(train_data, options);
    predicted_train = train_data * eigvector;
    predicted_test = test_data * eigvector;
    [OverallAccuracy, AverageAccuracy, Kappa] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, 'KNN'); %or â€˜SVMâ€?
    PCA_OA(i,:) = OverallAccuracy;
    PCA_AA(i,:) = AverageAccuracy;
    PCA_Kappa(i,:) = Kappa;
    %========================LPP================================
    [predicted_train, predicted_test] = run_LPP(train_data,test_data,1,1,30);
    [OverallAccuracy, AverageAccuracy, Kappa] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, 'KNN');
    LPP_OA(i,:) = OverallAccuracy;
    LPP_AA(i,:) = AverageAccuracy;
    LPP_Kappa(i,:) = Kappa;
    %========================GbFA================================
    [predicted_train, predicted_test] = run_GbFA(train_data,test_data,train_labels,5,1,30);
    [OverallAccuracy, AverageAccuracy, Kappa] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, 'KNN');
    GbFA_OA(i,:) = OverallAccuracy;
    GbFA_AA(i,:) = AverageAccuracy;
    GbFA_Kappa(i,:) = Kappa;
    %=========================FPCA================================
    [predicted_train, predicted_test] = run_FPCA(train_data,test_data,30);
    [OverallAccuracy, AverageAccuracy, Kappa] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, 'KNN');
    FPCA_OA(i,:) = OverallAccuracy;
    FPCA_AA(i,:) = AverageAccuracy;
    FPCA_Kappa(i,:) = Kappa;
    %=========================FLPP================================
    [predicted_train, predicted_test] = run_FLPP(train_data,test_data,1,0.1,30);
    [OverallAccuracy, AverageAccuracy, Kappa] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, 'KNN');
    FLPP_OA(i,:) = OverallAccuracy;
    FLPP_AA(i,:) = AverageAccuracy;
    FLPP_Kappa(i,:) = Kappa;
    %=========================SFLPP================================
    [predicted_train, predicted_test] = run_SFLPP(train_data,test_data,train_labels,1,1,30);
    [OverallAccuracy, AverageAccuracy, Kappa] = Get_accuracy(predicted_train, predicted_test, train_labels, test_labels, 'KNN');
    SFLPP_OA(i,:) = OverallAccuracy;
    SFLPP_AA(i,:) = AverageAccuracy;
    SFLPP_Kappa(i,:) = Kappa;
end
MeanOA(1,:) = mean(PCA_OA); MeanAA(1,:) = mean(PCA_AA); MeanKappa(1,:) = mean(PCA_Kappa);
MeanOA(2,:) = mean(LPP_OA); MeanAA(2,:) = mean(LPP_AA); MeanKappa(2,:) = mean(LPP_Kappa);
MeanOA(3,:) = mean(GbFA_OA); MeanAA(3,:) = mean(GbFA_AA); MeanKappa(3,:) = mean(GbFA_Kappa);
MeanOA(4,:) = mean(FPCA_OA); MeanAA(4,:) = mean(FPCA_AA); MeanKappa(4,:) = mean(FPCA_Kappa);
MeanOA(5,:) = mean(FLPP_OA); MeanAA(5,:) = mean(FLPP_AA); MeanKappa(5,:) = mean(FLPP_Kappa);
MeanOA(6,:) = mean(SFLPP_OA); MeanAA(6,:) = mean(SFLPP_AA); MeanKappa(6,:) = mean(SFLPP_Kappa);
StdOA(1,:) = std(PCA_OA); StdAA(1,:) = std(PCA_AA); StdKappa(1,:) = std(PCA_Kappa);
StdOA(2,:) = std(LPP_OA); StdAA(2,:) = std(LPP_AA); StdKappa(2,:) = std(LPP_Kappa);
StdOA(3,:) = std(GbFA_OA); StdAA(3,:) = std(GbFA_AA); StdKappa(3,:) = std(GbFA_Kappa);
StdOA(4,:) = std(FPCA_OA); StdAA(4,:) = std(FPCA_AA); StdKappa(4,:) = std(FPCA_Kappa);
StdOA(5,:) = std(FLPP_OA); StdAA(5,:) = std(FLPP_AA); StdKappa(5,:) = std(FLPP_Kappa);
StdOA(6,:) = std(SFLPP_OA); StdAA(6,:) = std(SFLPP_AA); StdKappa(6,:) = std(SFLPP_Kappa);