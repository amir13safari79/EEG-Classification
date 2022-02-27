function J_score = fisher_multi_dimensional(features, y_train)
    features_class0 = features(:, find(y_train == 0));
    features_class1 = features(:, find(y_train == 1));
    
    mu_0 = mean(features, 2);
    mu_1 = mean(features_class0, 2);
    mu_2 = mean(features_class1, 2);
    S_b = (mu_1 - mu_0)*(mu_1 - mu_0)' + (mu_2 - mu_0)*(mu_2 - mu_0)';
    
    S1 = 0;
    N1 = size(features_class0, 2);
    for i = 1:1:N1
        x_i  = features_class0(:,i);
        S1 = S1 + (1/N1)*(x_i - mu_1)*(x_i - mu_1)';
    end
    
    S2 = 0;
    N2 = size(features_class1, 2);
    for i = 1:1:N2
        x_i  = features_class1(:,i);
        S2 = S2 + (1/N2)*(x_i - mu_2)*(x_i - mu_2)';
    end
    S_w = S1 + S2;
    
    J_score = trace(S_b) / trace(S_w);

end