function [extracted_features,...
    statistical_features_size, ...
    frequency_features_size]= feature_extraction(input_data)

    Train_Features = [];
    statistical_features_size = 0;
    frequency_features_size = 0;
    channel_size = 28;
    last_feature_no = 0;
    
    for i = 1:1:size(input_data, 3)
        last_feature_no = 0;
        NewSig = input_data(:,1:channel_size, i);

        %%%%%%%%%%%%% Statistical features %%%%%%%%%%%%%
        % 1. Channel variance
        for j = 1:1:channel_size 
            extracted_features(last_feature_no+j,i) = var(NewSig(:,j));
        end
        last_feature_no = last_feature_no + channel_size;

        % 2. Channel skewness
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j,i) = skewness(NewSig(:,j));
        end
        last_feature_no = last_feature_no + channel_size;

        % 3. Channel kurtosis
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j,i) = kurtosis(NewSig(:,j));
        end
        last_feature_no = last_feature_no + channel_size;

        % 4. Two-to-two correlation of channels
        % overally 378 features:
        for j = 1:1:channel_size
            for k = j+1:1:channel_size
                extracted_features(last_feature_no+1, i) = corr(NewSig(:,j), NewSig(:,k));
                last_feature_no = last_feature_no + 1;
            end
        end

        % 5. entropy of channels:
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j,i) = entropy(NewSig(:,j));
        end
        last_feature_no = last_feature_no + channel_size;

        statistical_features_size = last_feature_no; 
        %%%%%%%%%%%%% Frequency features %%%%%%%%%%%%%
        fs = 100;
        % 1. channel mean-frequency
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j,i) = meanfreq(NewSig(:, j), fs);
        end
        last_feature_no = last_feature_no + channel_size;

        % 2. channel median-frequency
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j,i) = medfreq(NewSig(:, j), fs);
        end
        last_feature_no = last_feature_no + channel_size;

        % 3. channel obw(Occupied Bandwidth)
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j,i) = obw(NewSig(:, j), fs);
        end
        last_feature_no = last_feature_no + channel_size;

        % 4. Energy of 7 frequency-bands for each channel:
        % overally 196 features:

        % frequency band = theta-delta
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [2 8]);
        end
        last_feature_no = last_feature_no + channel_size;

        % frequency band = alpha
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [9 15]);
        end
        last_feature_no = last_feature_no + channel_size;

        % frequency band = beta1
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [16 22]);
        end
        last_feature_no = last_feature_no + channel_size;

        % frequency band = beta2
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [23 29]);
        end
        last_feature_no = last_feature_no + channel_size;

        % frequency band = gamma1
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [30 36]);
        end
        last_feature_no = last_feature_no + channel_size;

        % frequency band = gamma2
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [37 43]);
        end
        last_feature_no = last_feature_no + channel_size;

        % frequency band = gamma3
        for j = 1:1:channel_size
            extracted_features(last_feature_no+j, i) = bandpower(NewSig(:, j), fs, [44 50]);
        end
        last_feature_no = last_feature_no + channel_size;
    end

    frequency_features_size = last_feature_no - statistical_features_size;

end