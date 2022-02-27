%% In the name of ALLAH
% CI project: phase1


%% load data and plot some channels:
load('data/All_data')

train_size = size(x_train, 3);
test_size = size(x_test, 3);
channel_size = size(x_train, 2);
L = size(x_train, 1);

% plot signals
% choose channels fz, c5, cz, c6 and o1
% with channel number 3, 13, 16, 19, 27 respectively
% for plot:
ch_selected_number = [3, 13, 16, 19, 27];
ch_selected = ['Fz'; 'C5'; 'Cz'; 'C6'; 'O1'];


% right
tr = 8; % label for 8th data is 1
figure
for i = 1:1:5
    subplot(5,1,i)
    plot(x_train(:,ch_selected_number(i), tr))
    xlim([0,L]);
    title(['Right hand movement: Trial #', num2str(tr), ' channel:', ch_selected(i,:)]);
end

% left
tr = 24; % label for 24th data is 0
figure
for i = 1:1:5
    subplot(5,1,i)
    plot(x_train(:,ch_selected_number(i), tr))
    xlim([0,L]);
    title(['Left hand movement: Trial #', num2str(tr), ' channel:', ch_selected(i,:)]);
end

%% Feature Extraction from x_train with feaure_extraction.m
Train_Features = [];
statistical_features_size = 0;
frequency_features_size = 0;

[Train_Features, statistical_features_size, frequency_features_size] = ...
    feature_extraction(x_train);


%% Normalize Train_Features:
[Normalized_Train_Features,xPS] = mapminmax(Train_Features) ;

%% find best features according to Fisher Criterion(one dimensional)
% sort each feature for statistical and frequency features Separately.
train_statistical_features = Normalized_Train_Features(1:statistical_features_size,:);
train_frequency_features = Normalized_Train_Features(statistical_features_size+1:end,:);

% calculate J for both statistical and frequency features
statistical_features_J = zeros(statistical_features_size, 1);
frequency_features_J = zeros(frequency_features_size, 1);

%find index of Right and Left hand movement in train data:
right_indices = find(y_train == 1);
left_indices = find(y_train == 0);

for i = 1:1:statistical_features_size
    feature_row = train_statistical_features(i,:);
    right_feature_row = feature_row(right_indices);
    left_feature_row = feature_row(left_indices);

    % calculate Characteristic for finding J:
    mu0 = mean(feature_row);
    mu1 = mean(left_feature_row);
    mu2 = mean(right_feature_row);

    sigma1 = var(left_feature_row);
    sigma2 = var(right_feature_row);

    J = ((abs(mu0-mu1))^2 + (abs(mu0-mu2))^2) / (sigma1+sigma2);

    statistical_features_J(i) = J;
end

for i = 1:1:frequency_features_size
    feature_row = train_frequency_features(i,:);
    right_feature_row = feature_row(right_indices);
    left_feature_row = feature_row(left_indices);

    % calculate Characteristic for finding J:
    mu0 = mean(feature_row);
    mu1 = mean(left_feature_row);
    mu2 = mean(right_feature_row);

    sigma1 = var(left_feature_row);
    sigma2 = var(right_feature_row);

    J = ((abs(mu0-mu1))^2 + (abs(mu0-mu2))^2) / (sigma1+sigma2);

    frequency_features_J(i) = J;
end

% sort results and select top 40 features from them
% sort J_values:
sorted_statistical_features_J = sort(statistical_features_J, 'descend');
sorted_frequency_features_J = sort(frequency_features_J, 'descend');

% With a loop of different modes we test the number of features of each group
feature_divisions = [8 32;16 24;24 16;32 8];
selected_statistical_features_ind = [];
selected_frequency_features_ind = [];
selected_Train_Features = [];
J_score_best = 0; % initial J_score_best = 0

for i = 1:size(feature_divisions, 1)
    f_number_selected = feature_divisions(i,1);
    s_number_selected = feature_divisions(i,2);
    % sort J values for statistical and frequency features descending order
    % to find top <number_selected> values:
    statistical_boundary_tmp = sorted_statistical_features_J(s_number_selected+1);
    frequency_boundary_tmp = sorted_frequency_features_J(f_number_selected+1);

    selected_statistical_features_ind_tmp = ...
        find(statistical_features_J > statistical_boundary_tmp);

    selected_frequency_features_ind_tmp = ...
        find(frequency_features_J > frequency_boundary_tmp);

    tr_selected_statistical_features_tmp = ...
        train_statistical_features(selected_statistical_features_ind_tmp,:);
    tr_selected_frequency_features_tmp = ...
        train_frequency_features(selected_frequency_features_ind_tmp, :);
    
    % merge two temporary selected features
    selected_Train_Features_tmp = [tr_selected_statistical_features_tmp;...
                                    tr_selected_frequency_features_tmp];
                                
    % find J_score with multi-dimensional mode:
    J_score = fisher_multi_dimensional(selected_Train_Features_tmp, y_train);
    
    % find best_J_score:
    if J_score > J_score_best
        J_score_best = J_score;
        selected_statistical_features_ind = selected_statistical_features_ind_tmp;
        selected_frequency_features_ind = selected_frequency_features_ind_tmp;
        selected_Train_Features = selected_Train_Features_tmp;
    end
end

%% implement MLP classifier
max_neuron = 10; % per layer
activation_functions = ["radbas"; "logsig"; "purelin"; "satlin"; "tansig"; "hardlims"];
activation_size = numel(activation_functions); 
mlp_ACCMat = zeros(max_neuron, activation_size);

% with 1 layer hidden size
for n1 = 1:max_neuron
    for active_ind = 1:1:activation_size
        ACC = 0 ; 
        % 5-fold cross-validation
        fold_size = round(train_size/5);
        for k =1:5
            if k<5
                train_indices = [1:(k-1)*fold_size,k*fold_size+1:train_size] ;
                valid_indices = (k-1)*fold_size+1:k*fold_size ;
            else
                train_indices = 1:4*fold_size;
                valid_indices = 4*fold_size+1:train_size;
            end

            TrainX = selected_Train_Features(:,train_indices) ;
            ValX = selected_Train_Features(:,valid_indices) ;
            TrainY = y_train(train_indices) ;
            ValY = y_train(valid_indices) ;

            % feedforwardnet, newff, paternnet
            % net = patternnet(hiddenSizes,trainFcn,performFcn)
            net = patternnet(n1, 'trainbr', 'mse');
            net = train(net,TrainX,TrainY);
            net.layers{2}.transferFcn = activation_functions(active_ind);
            
            predict_y = net(ValX);

            % find treshhold
            p_TrainY = net(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        mlp_ACCMat(n1, active_ind) = ACC/train_size;
    end
end

% find the best neuron size and activation function:
[Amaxs, idx] = max(mlp_ACCMat);
[Amax, Aj] = max(Amaxs);
Ai = idx(Aj);
mlp_best_acc = Amax;
mlp_best_neuron_size = Ai;
mlp_best_active_func_ind = Aj;
mlp_best_active_func = activation_functions(Aj);

%% implement RBF classifier
spreadMat = 1:0.3:5 ;
spreadMat_size = size(spreadMat,2);
NMat = [2,5,7,10,12,15,17,20,22,25,27];
NMat_size = size(NMat, 2);
rbf_ACCMat = zeros(spreadMat_size, NMat_size);

for s = 1:spreadMat_size
    spread = spreadMat(s) ;
    for n = 1:NMat_size 
        Maxnumber = NMat(n) ;
        ACC = 0 ;
        % 6-fold cross-validation
        fold_size = round(train_size/5);
        for k =1:5
            if k<5
                train_indices = [1:(k-1)*fold_size,k*fold_size+1:train_size] ;
                valid_indices = (k-1)*fold_size+1:k*fold_size ;
            else
                train_indices = 1:4*fold_size;
                valid_indices = 4*fold_size+1:train_size;
            end

            TrainX = selected_Train_Features(:,train_indices) ;
            ValX = selected_Train_Features(:,valid_indices) ;
            TrainY = y_train(train_indices) ;
            ValY = y_train(valid_indices) ;

            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
            predict_y = net(ValX);
            
            % find treshhold
            Thr = 0.5;

            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        rbf_ACCMat(s,n) = ACC/train_size;
    end
end

% find the best spreadMat and NMat:
[Amaxs, idx] = max(rbf_ACCMat);
[Amax, Aj] = max(Amaxs);
Ai = idx(Aj);
rbf_best_acc = Amax;
rbf_best_spreadMat_ind = Ai;
rbf_best_spread = spreadMat(Ai);
rbf_best_NMat_ind = Aj;
rbf_best_N = NMat(Aj);

%% classify test data(x_test) with best parameters
% with MLP network and RBF :
% find Test_Features with feature_extraction.m
[Test_Features,~,~] = feature_extraction(x_test);

% Normalize Test_Features:
Normalized_Test_Features = mapminmax('apply', Test_Features, xPS);

test_statistical_features = Normalized_Test_Features(1:statistical_features_size,:);
test_frequency_features = Normalized_Test_Features(statistical_features_size+1:end,:);

te_selected_statistical_features = ...
    test_statistical_features(selected_statistical_features_ind,:);
te_selected_frequency_features = ...
    test_frequency_features(selected_frequency_features_ind, :);

% finally merge two selected features:
selected_Test_Features = [te_selected_statistical_features; te_selected_frequency_features];

%% Classify with MLP:
N = mlp_best_neuron_size;
activation_func = mlp_best_active_func;

TrainX = selected_Train_Features;
TrainY = y_train;
TestX = selected_Test_Features;

net = patternnet(N, 'trainbr', 'mse');
net = train(net,TrainX,TrainY);
net.layers{2}.transferFcn = activation_func;

mlp_predict_y = net(TestX);

% find treshhold
p_TrainY = net(TrainX);
[X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

% find label for x_test
mlp_predict_y_test = mlp_predict_y >= Thr ;

% save result:
save('mlp_predict_y_test.mat','mlp_predict_y_test');

%% Classify with RBF:
spread = rbf_best_spread;
Maxnumber = rbf_best_N;

TrainX = selected_Train_Features;
TrainY = y_train;
TestX = selected_Test_Features;

net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
rbf_predict_y = net(TestX);

% set treshhold = 0.5
Thr = 0.5 ;
rbf_predict_y_test = rbf_predict_y >= Thr ;

% save result:
save('rbf_predict_y_test.mat','rbf_predict_y_test');

%% save selected train and test features for using in phase2
save('selected_Train_Features.mat', 'selected_Train_Features');
save('selected_Test_Features.mat', 'selected_Test_Features');