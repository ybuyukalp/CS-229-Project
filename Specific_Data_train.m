% Abnormal Combustion Detection
% CS 229 - Final Project
clear all; close all; clc;
% Load the structered experiment data
load('Data_File.mat');
Label = importdata('Small_Data_Labels.xlsx');
Label = Label(:,1)
% Extract specific the Pressure Trace and Injection Information
a=fieldnames(Data);
P_ci_comb_all = [];
CA_injection  = [];
names_of_empty_pcicomb = {};
names_of_empty_Inj_CAi    = {};
for i=1:length(a)
    b = getfield(Data,a{i});
    if ~isempty(b.P_ci_comb)
        P_ci_comb_all = [P_ci_comb_all; b.P_ci_comb(end,:)];
        CA_injection  = [CA_injection; b.Inj_CAi(:,:)];
    else
        names_of_empty_pcicomb  = [names_of_empty_pcicomb; a{i}];
        names_of_empty_Inj_CAi     = [names_of_empty_Inj_CAi; a{i}];
    end
end
% Plot the extracted data to see if it worked
figure(1);
for i = 1:size(P_ci_comb_all, 1)
    plot(P_ci_comb_all(i, :));
    hold on;
end
%% This section shows examples of normal and abnormal pressure trace
crank_angle = linspace(-360,360,7200);
figure(2);
plot(crank_angle,P_ci_comb_all(7,:),'LineWidth',1.5);
hold on;
plot(crank_angle,P_ci_comb_all(4,:),'LineWidth',1.5);
legend('Normal','Abnormal')
ylabel('In-Cylinder Pressure (bar)');
xlabel('Crank Angle (\circ)');
xlim([-30 30]);
ylim([10 55]);
set(gca, 'FontSize', 18);
%% Here we separete the training data and the test data
% Construct training data for pressure
Ptrain = P_ci_comb_all(1:5,:); 
Ptrain = cat(1,Ptrain,P_ci_comb_all(7:16,:));
Ptrain = cat(1,Ptrain,P_ci_comb_all(23:29,:));
% Construct test data for pressure
Ptest  = cat(1,P_ci_comb_all(6,:),P_ci_comb_all(17:22,:));
num_data = size(Ptrain, 1); % Number of training data
% Allocate Space for gradients
array_max_FX = zeros(1, num_data);
array_min_FX = zeros(1, num_data);
array_index_max_FX = zeros(1, num_data);
array_index_min_FX = zeros(1, num_data);
crank_angle = linspace(-360,360,7200);
% Calculate gradients for the training data
for i = 1:num_data
    FX = gradient(Ptrain(i, :),crank_angle(2)-crank_angle(1));
    array_max_FX(i) = max(FX);
    array_min_FX(i) = min(FX);
    dummy = find(FX == max(FX));
    array_index_max_FX(i) = dummy(1);
    dummy = find(FX == min(FX));
    array_index_min_FX(i) = dummy(1);
end
% Calculate gradients for the test data
num_test = size(Ptest,1); % Number of test data
for i = 1:num_test
    FX_test = gradient(Ptest(i, :), crank_angle(2)-crank_angle(1));
    array_max_FX_test(i) = max(FX_test);
    array_min_FX_test(i) = min(FX_test);
    dummy = find(FX_test == max(FX_test));
    array_index_max_FX_test(i) = dummy(1);
    dummy = find(FX_test == min(FX_test));
    array_index_min_FX_test(i) = dummy(1);
end
%% Injection Timings and Injections Durations for All
% Allocate space for the injection data
injTiming   = zeros(size(CA_injection,1), 1);
injDuration = zeros(size(CA_injection,1), 1);
% Calculate injection timing and injection durations
for i = 1:size(CA_injection,1)
    injTiming(i,1)   = CA_injection(i,2);
    injDuration(i,1) = abs(CA_injection(i,4) - CA_injection(i,3));
end
%% Here we separete the training data and the test data for injections
injT_train   = cat(1,injTiming(1:5,1),injTiming(7:16,1));
injT_train   = cat(1,injT_train,injTiming(23:29,1));
injT_test    = cat(1,injTiming(6,1),injTiming(17:22,1));
%% Train and Test the data
% Create the test arrays
xdata_test(:,1) = abs(array_min_FX_test); % Absolute value of the minimum
xdata_test(:,2) = abs(array_max_FX_test); % Maimum pressure gradient
% Create inputs for the train data
xdata(:,1) = abs(array_min_FX); % Absolute value of the minimum
xdata(:,2) = abs(array_max_FX); % Maimum pressure gradient
X = xdata; % Rows: Experiments, Columns: Pressure Gradients
k = 2; % Obtain 2 groups
% Apply k-means (2-means) to the data for labeling
idx = kmeans(X,k); 
% Plot resulting clusters
figure;
for i=1: size(xdata,1)
    if(idx(i) == 1)
        % NORMAL COMBUSTION
        plot(xdata(i,1), xdata(i,2), 'b*');
    else
        % ABNORMAL COMBUSTION
        plot(xdata(i,1), xdata(i,2), 'r*');
    end
    hold on;
end
% Include the test data on the same plot
plot(xdata_test(:,1),xdata_test(:,2),'gd')
xlabel('(dP/dt)_{min}')
ylabel('(dP/dt)_{max}')
plotfixer
%% Apply Support Vector Machines
% Rename the Labels
groups = idx;
%% 
% Apply and Plot the Support Vectors and Test 
figure;
clf; hold on
plot(xdata_test(:,1),xdata_test(:,2),'gd')
svmStruct = svmtrain(xdata,groups,'ShowPlot',true,'AutoScale',true);
xlabel('(dP/dt)_{min}')
ylabel('(dP/dt)_{max}')
plotfixer
%%
figure(100); 
hold on
plot(crank_angle,Ptest(2,:))
plot(crank_angle,Ptest(6,:))
xlabel('Crank Angle Degree')
ylabel('Pressure (bar)')
grid
xlim([-30 30])
ylim([20 70])
plotfixer
%% 
FXnormal = gradient(Ptrain(7, :), crank_angle(2)-crank_angle(1));
FXabnormal = gradient(Ptrain(4, :), crank_angle(2)-crank_angle(1));
FXXn = gradient(FXnormal, crank_angle(2)-crank_angle(1));
FXXa = gradient(FXabnormal, crank_angle(2)-crank_angle(1));
figure(101)
hold on
plot(crank_angle,FXnormal,'-','LineWidth',1)
plot(crank_angle,FXabnormal,'-','LineWidth',1)
legend('Normal','Abnormal')
xlabel('Crank Angle Degree')
ylabel('dP/dCA')
xlim([0 30])
plotfixer
