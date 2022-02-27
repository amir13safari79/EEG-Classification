%% In the name of ALLAH
% CI project: phase2


%% load selected Train and Test features and y_train
load('data/All_data')
load('selected_Train_Features.mat')
load('selected_Test_Features.mat')

%% find genetic_features(features that selected with genetic algorithm)
% use fisher_multi_dimensional as fitness function
nvars = 40; % number of variables
FunctionTolerance = 1e-3;
options = optimoptions('ga');
options = optimoptions(options,'PopulationType', 'bitstring');
options = optimoptions(options,'FunctionTolerance', FunctionTolerance);
options = optimoptions(options,'FitnessScalingFcn', @fitscalingprop);
options = optimoptions(options,'SelectionFcn', @selectionroulette);
options = optimoptions(options,'CrossoverFcn', @crossovertwopoint);
options = optimoptions(options,'MutationFcn', {  @mutationuniform [] });
options = optimoptions(options,'Display', 'final');
options = optimoptions(options,'PlotFcn', {  @gaplotbestf @gaplotbestindiv @gaplotscores @gaplotstopping });
[genetic_ind,fval,exitflag,output,population,score] = ...
ga(@fisher_fitness,nvars,[],[],[],[],[],[],[],[],options);

% find genetic_Train_Featuers 
% and genetic_Test_Features according to genetic_ind
genetic_Train_Features = selected_Train_Features(find(genetic_ind),:);
genetic_Test_Features = selected_Test_Features(find(genetic_ind),:);
train_size = size(genetic_Train_Features, 2);

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

            TrainX = genetic_Train_Features(:,train_indices) ;
            ValX = genetic_Train_Features(:,valid_indices) ;
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
genetic_mlp_best_acc = Amax;
genetic_mlp_best_neuron_size = Ai;
genetic_mlp_best_active_func_ind = Aj;
genetic_mlp_best_active_func = activation_functions(Aj);

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

            TrainX = genetic_Train_Features(:,train_indices) ;
            ValX = genetic_Train_Features(:,valid_indices) ;
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
genetic_rbf_best_acc = Amax;
genetic_rbf_best_spreadMat_ind = Ai;
genetic_rbf_best_spread = spreadMat(Ai);
genetic_rbf_best_NMat_ind = Aj;
genetic_rbf_best_N = NMat(Aj);

%% classify genetic_Test_Features with best parameters:
% Classify with MLP:
N = genetic_mlp_best_neuron_size;
activation_func = genetic_mlp_best_active_func;

TrainX = genetic_Train_Features;
TrainY = y_train;
TestX = genetic_Test_Features;

net = patternnet(N, 'trainbr', 'mse');
net = train(net,TrainX,TrainY);
net.layers{2}.transferFcn = activation_func;

genetic_mlp_predict_y = net(TestX);

% find treshhold
p_TrainY = net(TrainX);
[X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

% find label for x_test
genetic_mlp_predict_y_test = genetic_mlp_predict_y >= Thr ;

% save result:
save('genetic_mlp_predict_y_test.mat','genetic_mlp_predict_y_test');

%% Classify with RBF:
spread = genetic_rbf_best_spread;
Maxnumber = genetic_rbf_best_N;

TrainX = genetic_Train_Features;
TrainY = y_train;
TestX = genetic_Test_Features;

net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
genetic_rbf_predict_y = net(TestX);

% set treshhold = 0.5
Thr = 0.5 ;
genetic_rbf_predict_y_test = genetic_rbf_predict_y >= Thr ;

% save result:
save('genetic_rbf_predict_y_test.mat','genetic_rbf_predict_y_test');

