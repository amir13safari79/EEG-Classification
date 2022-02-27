function score = fisher_fitness(x)
    % load selected_Train_Features that obtained in phase1
    load('selected_Train_Features.mat');
    
    % load y_train:
    load('All_data.mat')
    
    % find features according to x Chromosome
    ind = find(x == 1);
    features = selected_Train_Features(ind,:);
    p = numel(ind);  % p is number of features
    
    % apply Fisher Criterion(multi dimensional)
    J_score = fisher_multi_dimensional(features, y_train);
    
    if (sum(x) > 15 && sum(x) < 30)
        score = -(J_score);
    elseif(sum(x) <= 15)
        score = -(J_score) + (15 - sum(x));
    else
        score = -(J_score) + (sum(x) - 30);
    end

end