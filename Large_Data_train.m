% Abnormal Combustion Detection
% CS 229 - Final Project
clear all; close all; clc;
% Load the structered experiment data
load('Data_File.mat');
Label = importdata('Small_Data_Labels.xlsx');
Label = Label(:,1);
a=fieldnames(Data);
P_ci_comb_all = [];
CA_injection  = [];
names_of_empty_pcicomb = {};
names_of_empty_Inj_CAi = {};
iter = 1;
for i=1:length(a)
    b = getfield(Data,a{i});
    if ~isempty(b.P_ci_comb)
        P_ci_comb_all               = [P_ci_comb_all; b.P_ci_comb];
        CA_injection                = [CA_injection; b.Inj_CAi(:,:)];
        CountData(i,1) = size(b.P_ci_comb,1);
    else
        names_of_empty_pcicomb      = [names_of_empty_pcicomb; a{i}];
        names_of_empty_Inj_CAi      = [names_of_empty_Inj_CAi; a{i}];
    end
end
% Extract Injection Information
Injection = zeros(135,6);
sum = 0;
getNumData = CountData(CountData>0);
for i = 1:length(getNumData)
    for j = (sum+1):(sum+1)+getNumData(i)
        Injection(j,:) = CA_injection(i,:);
        if(rem(j,3) == 1 && j < 135)
            testP(j,:) = P_ci_comb_all(j,:);
        elseif (j <= 135)
            trainP(j,:)  = P_ci_comb_all(j,:);
        end
    end
    sum = sum + getNumData(i);  
end
Injection = Injection(1:end-1,:);
%% Get other parameters, injection duration, timing, pressure gradients
%  Also, separate training data and the test data
numData = 120;
Ptrain = P_ci_comb_all(1:numData,:);
Ptest  = P_ci_comb_all(numData+1:end,:);
crank_angle = linspace(-360,360,7200);
array_max_FX = zeros(1, numData);
array_min_FX = zeros(1, numData);
array_index_max_FX = zeros(1, numData);
array_index_min_FX = zeros(1, numData);
% Get pressure gradients
for i = 1:numData
    FX = gradient(Ptrain(i, :), crank_angle(2)-crank_angle(1));
    array_max_FX(i) = max(FX);
    array_min_FX(i) = min(FX);
    dummy = find(FX == max(FX));
    array_index_max_FX(i) = dummy(1);
    dummy = find(FX == min(FX));
    array_index_min_FX(i) = dummy(1);
end
% Get the gradients for the test data
numTest = size(Ptest,1);
array_max_FX_test = zeros(1, numTest);
array_min_FX_test = zeros(1, numTest);
array_index_max_FX_test = zeros(1, numTest);
array_index_min_FX_test = zeros(1, numTest);
for i = 1:numTest
    FX_test = gradient(Ptest(i, :), crank_angle(2)-crank_angle(1));
    array_max_FX_test(i) = max(FX_test);
    array_min_FX_test(i) = min(FX_test);
    dummy = find(FX_test == max(FX_test));
    array_index_max_FX_test(i) = dummy(1);
    dummy = find(FX_test == min(FX_test));
    array_index_min_FX_test(i) = dummy(1);
end
%% Get Injetion Information
%  Calculate Injection Timings and Injections Durations
injTiming   = zeros(size(Injection,1), 1);
injDuration = zeros(size(Injection,1), 1);
for i = 1:size(Injection,1)
    injTiming(i,1)   = Injection(i,2);
    injDuration(i,1) = abs(Injection(i,4) - Injection(i,3));
end
% Separate injection timing test and train data
injT_train   = injTiming(1:numData)
injT_test    = injTiming(numData+1:end,:)
% Separate injection duration test and train data
injD_train   = injDuration(1:numData)
injD_test    = injDuration(numData+1:end,:)
%% Obtain the parameters to feed into k-means
xdata3(:,1) = injT_train;
xdata3(:,2) = abs(array_min_FX);
xdata3(:,3) = abs(array_max_FX);
X3 = xdata3;
%% Obtain the test data to compare with the k-means
xtest3(:,1) = injT_test;
xtest3(:,2) = abs(array_min_FX_test);
xtest3(:,3) = abs(array_max_FX_test);
X3test = xtest3;
%% Apply 3D k-means
k = 3;
[Groups,Centroids] = kmeans(X3,k);
% show points and clusters (color-coded)
clr = lines(k);
figure, hold on
scatter3(X3(:,1), X3(:,2), X3(:,3), 72, clr(Groups,:), 'Marker','*')
scatter3(Centroids(:,1), Centroids(:,2), Centroids(:,3), 100, clr, 'Marker','o', 'LineWidth',5)
grid on
hold off
view(3), axis vis3d, box on, rotate3d on
xlabel('Injection Timing'), ylabel('(dp/dt)_{min}'), zlabel('(dp/dt)_{max}')
plotfixer
%%
% Create the train data
xAll(:,2) = abs(array_min_FX);
xAll(:,3) = abs(array_max_FX);
xAll(:,1) = injT_train;
xdata = xAll;
randVec = randi([1 120],[1 15])
xtest   = zeros(15,2)
for i = 1:15
   xtest(i,2) = xAll(randVec(i),2)
   xtest(i,3) = xAll(randVec(i),3)
   xtest(i,1) = xAll(randVec(i),1)
end
% xdata   = zeros(15,2)
for i = 1:15
   xdata(randVec(i),2) = 0;
   xdata(randVec(i),3) = 0;
   xdata(randVec(i),1) = 1000;
end
%%
x2 = xdata(:,2);
x2 = x2(x2>0);
xtrain(:,2) = x2;
x3 = xdata(:,3);
x3 = x3(x3>0);
xtrain(:,3) = x3;
x1 = xdata(:,1);
x1 = x1(x1<1000);
xtrain(:,1) = x1;
Xtrain = xtrain;
Xtest  = xtest;
%%
k = 2;
[Groups,Centroids] = kmeans(Xtrain,k);
%%
%# show points and clusters (color-coded)
clr = lines(k);
figure, hold on
scatter3(Xtrain(:,1), Xtrain(:,2), Xtrain(:,3), 72, clr(Groups,:), 'Marker','*')
scatter3(Xtest(:,1), Xtest(:,2), Xtest(:,3), 72, 'Marker','d')
scatter3(Centroids(:,1), Centroids(:,2), Centroids(:,3), 100, clr, 'Marker','o', 'LineWidth',5)
grid on
hold off
view(3), axis vis3d, box on, rotate3d on
xlabel('Injection Timing'), ylabel('(dp/dt)_{min}'), zlabel('(dp/dt)_{max}')
plotfixer
%% Train all data in 2-D
% Label the data
xdata1(:,1) = Xtrain(:,2);
xdata1(:,2) = Xtrain(:,3);
X = xdata1;
k = 2;
idx = kmeans(X,k);
figure;
for i=1: size(xdata1,1)
    if(idx(i) == 1)
        plot(xdata1(i,1), xdata1(i,2), 'b*');
    else
        plot(xdata1(i,1), xdata1(i,2), 'r*');
    end
    hold on;
end      
plot(Xtest(:,2),Xtest(:,3),'kd')
%% APPLY SVM AFTER THE LABELS
figure; 
svmStruct = svmtrain(xdata1,idx,'ShowPlot',true);
hold on;
plot(Xtest(:,2),Xtest(:,3),'diamond','Color','green');
legend('Normal','Abnormal','Support Vectors','SVM','Test');
xlabel('(dP/dt)_{min}')
ylabel('(dP/dt)_{max}')
plotfixer
%% SHOW 3D PLOT
points = [0 30.15;30.7155 0]
X = Xtrain
clr = lines(k);
Groups = idx;
figure, clf; hold on
scatter3(X(:,1), X(:,2), X(:,3), 72, clr(Groups,:), 'Marker','*')
scatter3(Xtest(:,1), Xtest(:,2), Xtest(:,3), 72, 'Marker','d')
patch( [-25 0 0 -25], [points(1,2) points(1,2) 0 0], [0 0 points(2,1) points(2,1)],[0 1 2 3]);
legend('Train','Test','SVM')
grid on;
hold off;
xlabel('Injection Timing - BTDC (CAD^{\circ})'), ylabel('(dp/dt)_{min}'), zlabel('(dp/dt)_{max}');
plotfixer;

