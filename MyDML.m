function out = MyDML(dataName, initDataX, initDataY, drMethod)

nSamples = size(initDataX, 1);
D = size(initDataX, 2); % The ambient space dimensionality.

if D > nSamples
    
    D = nSamples;
    
end

interval = floor(D / 5) + 1;

nFolds = 10; % Number of folds used in the k-fold cross-vlaidation.

% result.KNN.CP = [];
result.KNN.AvgSen = [];
result.KNN.AvgSpec = [];
result.KNN.AvgAcc = [];
result.KNN.AvgTime = [];
result.KNN.AvgCM = [];

% result.simKNN.CP = [];
result.simKNN.AvgSen = [];
result.simKNN.AvgSpec = [];
result.simKNN.AvgAcc = [];
result.simKNN.AvgTime = [];
result.simKNN.AvgCM = [];

% result.SVM.CP = [];
result.SVM.AvgSen = [];
result.SVM.AvgSpec = [];
result.SVM.AvgAcc = [];
result.SVM.AvgTime = [];
result.SVM.AvgCM = [];

result.d = [];  % Maximum embedding dimensionality based on the variance.

for d = 1 : interval : D
    
    KNNSpec = zeros(nFolds, 1);
    simKNNSpec = zeros(nFolds, 1);
    SVMSpec = zeros(nFolds, 1);
    
    KNNSen = zeros(nFolds, 1);
    simKNNSen = zeros(nFolds, 1);
    SVMSen = zeros(nFolds, 1);
    
    KNNAcc = zeros(nFolds, 1);
    simKNNAcc = zeros(nFolds, 1);
    SVMAcc = zeros(nFolds, 1);
    
    KNNTime = zeros(nFolds, 1);
    simKNNTime = zeros(nFolds, 1);
    SVMTime = zeros(nFolds, 1);
    
    SVMConf = [];
    simKNNConf = [];
    KNNConf = [];
    
    for ii = 1 : nFolds
        
        if strcmp(dataName, 'KDD')
            
            reductionPerc = 0.01; % Sampling precentage
            [initTrX, trY, initTstX, tstY] = ...
                MyCVPartition(nFolds, initDataX, initDataY, reductionPerc);
            
        else
            
            reductionPerc = 1; % Sampling precentage
            [initTrX, trY, initTstX, tstY] = ...
                MyCVPartition(nFolds, initDataX, initDataY, reductionPerc);
            
        end
        
        nTr = size(initTrX, 1);
        nTst = size(initTstX, 1);
        nCalss = numel(unique(trY));
        
        
        %% Calculate the embeded neighborhoods using the LLE transform.
        
        K = 10; % Number of the nearset neighbors to be used on the manifold.
        
        [sampledX, sampledY] = MySampling(initTrX, trY);
        
        nSampled = size(sampledX, 1);
        
        if strcmp(drMethod,'LDA')
            
            [emTrX, mapping] = compute_mapping([sampledY, sampledX], drMethod, d);    % Embeded training data.
            
        else
            
            [emTrX, mapping] = compute_mapping(sampledX, drMethod, d);    % Embeded training data.
            
        end
        
        if size(emTrX, 1) ~= size(sampledX, 1)
            
            emTrX = out_of_sample(sampledX, mapping);
            
        end
        
        
        %% Find the k-Nearest and farest Neighbors from the neighborhoods on the Manifold.
        
        nPointsPatch = 7;  % Number of the nearest neighbors to be considered on a patch.
        
        distMat = CalcDist(emTrX);
        
        % Sort neighbors according to their Euclidean distance from the centric
        % data-point on each patch.
        [~, sortOrder] = sort(distMat, 2); %  Sorts the elements of each row.
        
        nnInd = sortOrder(:, 2 : end);      % Nearest Neighbors Index.
        fnInd = sortOrder(:, end: -1 : 2);  % Farest Neighbors Index.
        
        %% Create the similar and dissimilar matirx.
        
        Y = zeros(nSampled);
        sdMat = zeros(nSampled);    % Similar/dissimilar/unrealted matirx.
        
        nnSimInd = zeros(nSampled, nPointsPatch);
        nnDisInd = zeros(nSampled, nPointsPatch);
        
        for i = 1 : nSampled
            
            j = 1;
            nSim = 0;
            nDissim = 0;
            simInd = [];
            dissimInd = [];
            
            % Create the similar and dissimlar patches.
            while (nSim < nPointsPatch + 1) || (nDissim < nPointsPatch + 1)
                
                if j > size(nnInd, 2)
                    
                    break;
                    
                end
                
                if (sampledY(i) == sampledY(fnInd(i, j)))
                    
                    simInd = [simInd, fnInd(i, j)];
                    nSim = nSim + 1;
                    Y(i, simInd) = 1;
                    
                end
                
                if (sampledY(i) ~= sampledY(nnInd(i, j)))
                    
                    dissimInd = [dissimInd, nnInd(i, j)];
                    nDissim = nDissim + 1;
                    
                end
                
                j = j + 1;
                
            end
            
            nnSimInd(i, :) = simInd(1 : nPointsPatch);
            nnDisInd(i, :) = dissimInd(1 : nPointsPatch);
            
            sdMat(i, nnSimInd(i, :)) = +1;  % Similar
            sdMat(i, nnDisInd(i, :)) = -1;  % Dissimilar
            
        end
        
        
        %% Distance Metric Learning using DLSR.
        
        maxIt = 30;
        
        B=sdMat;
        M = zeros(size(B));
        W0 = zeros(D, size(Y,2));
        t0 = zeros(size(Y,2), 1);
        
        en = ones(nSampled, 1);
        H = eye(nSampled) - (1/nSampled) * (en * en');
        lambda = 1; % Positive Regularization Parameter.
        
        U = (sampledX' * H * sampledX + lambda * eye(D))\ (sampledX' * H);
        
        for it = 1 : maxIt
            R = Y + B .* M;
            W = U * R;
            t = (1/nSampled) * R' * en - (1/nSampled) * W' * sampledX' * en;
            P = sampledX * W + en * t' - Y;
            M = max(B .* P, 0);
            
            if (norm(W-W0, 'fro')^2 + norm(t-t0, 2)^2) < 10^(-4)
                
                break;
                
            end
            
            W0 = W;
            t0 = t;
            
        end
        
        % Calculate the images of the data points under the above linear
        % transformation.
        mappedTrX = initTrX * W + repmat(t, 1, nTr)';
        
        
        %% Creat Mapped test data.
        
        mappedTst = initTstX * W + repmat(t, 1, nTst)';
        
        
        %% Apply the dist-k-NN classifier
        
        nNN = 7; % Number of nearest neighbors.
        nnLabels = zeros(nTst, nNN);
        KNNOut = zeros(nTst, 1);
        
        tic;
        for i = 1 : nTst
            
            nnInd = FindKNN(mappedTst(i, :), mappedTrX, nNN);
            nnLabels(i, :) = trY(nnInd);
            KNNOut(i) = mode(nnLabels(i, :));
            
        end
        KNNTime(ii) = toc / nTst;
        
        CP_KNN = classperf(tstY, KNNOut);
        
        [currentKNNConf, order]  = confusionmat(tstY, KNNOut); %  Confusion Matrix of the k-NN
        
        KNNSpec(ii) = CP_KNN.Specificity;
        KNNSen(ii) = CP_KNN.Sensitivity;
        KNNAcc(ii) = CP_KNN.CorrectRate;
        KNNConf = cat(3, KNNConf, currentKNNConf);
        
        %% Apply the sim-k-NN classifier
        
        nNN = 7; % Number of nearest neighbors.
        nnLabels = zeros(nTst, nNN);
        simKNNOut = zeros(nTst, 1);
        
        [~, nnInd] = sort(mappedTst, 2, 'descend');
        
        tic;
        for i = 1 : nTst
            
            nnLabels(i, :) = trY(nnInd(i, 1 : nNN));
            simKNNOut(i) = mode(nnLabels(i, :));
            
        end
        simKNNTime(ii) = toc / nTst;
        
        CP_simKNN = classperf(tstY, simKNNOut);
        
        [currentSimKNNConf, order]  = confusionmat(tstY, simKNNOut); %  Confusion Matrix of the k-NN
        
        simKNNSpec(ii) = CP_simKNN.Specificity;
        simKNNSen(ii) = CP_simKNN.Sensitivity;
        simKNNAcc(ii) = CP_simKNN.CorrectRate;
        simKNNConf = cat(3, simKNNConf, currentSimKNNConf);
        
        
        %% Apply the SVM Classifer
        
        tic;
        SVMOut = MultiSVM(mappedTrX, trY, mappedTst);
        SVMTime(ii) = toc / nTst;
        
        CP_SVM = classperf(tstY, SVMOut);
        
        [currentSVMConf, order]  = confusionmat(tstY, SVMOut); %  Confusion Matrix of SVM
        
        SVMSpec(ii) = CP_SVM.Specificity;
        SVMSen(ii) = CP_SVM.Sensitivity;
        SVMAcc(ii) = CP_SVM.CorrectRate;
        SVMConf = cat(3, SVMConf, currentSVMConf);
        
        % Display Test Results
        disp(['Iteration ', num2str(ii), ...
            ': SVM Spec = ', num2str(SVMSpec(ii)), ...
            ', Sen = ', num2str(SVMSen(ii)), ...
            ', Acc = ', num2str(SVMAcc(ii)), ...
            ', Time = ', num2str(SVMTime(ii)), ...
            ': k_NN Spec = ', num2str(KNNSpec(ii)), ...
            ', Sen = ', num2str(KNNSen(ii)), ...
            ', Acc = ', num2str(KNNAcc(ii)), ...
            ', Time = ', num2str(KNNTime(ii)), ...
            ': sim-k_NN Spec = ', num2str(simKNNSpec(ii)), ...
            ', Sen = ', num2str(simKNNSen(ii)), ...
            ', Acc = ', num2str(simKNNAcc(ii)), ...
            ', Time = ', num2str(simKNNTime(ii))]);
        
        
    end
    
    result.KNN.AvgSen = [result.KNN.AvgSen, mean(KNNSen)];
    result.KNN.AvgSpec = [result.KNN.AvgSpec, mean(KNNSpec)];
    result.KNN.AvgAcc = [result.KNN.AvgAcc, mean(KNNAcc)];
    result.KNN.AvgTime = [result.KNN.AvgTime, mean(KNNTime)];
    result.KNN.AvgCM = [result.KNN.AvgCM, mean(KNNConf, 3)];
    
    result.simKNN.AvgSen = [result.simKNN.AvgSen, mean(simKNNSen)];
    result.simKNN.AvgSpec = [result.simKNN.AvgSpec, mean(simKNNSpec)];
    result.simKNN.AvgAcc = [result.simKNN.AvgAcc, mean(simKNNAcc)];
    result.simKNN.AvgTime = [result.simKNN.AvgTime, mean(simKNNTime)];
    result.simKNN.AvgCM = [result.simKNN.AvgCM, mean(simKNNConf, 3)];
    
    result.SVM.AvgSen = [result.SVM.AvgSen, mean(SVMSen)];
    result.SVM.AvgSpec = [result.SVM.AvgSpec, mean(SVMSpec)];
    result.SVM.AvgAcc = [result.SVM.AvgAcc, mean(SVMAcc)];
    result.SVM.AvgTime = [result.SVM.AvgTime, mean(SVMTime)];
    result.SVM.AvgCM = [result.SVM.AvgCM, mean(SVMConf, 3)];
    
    result.d = [result.d, d];
    
end


%% Show results

% Accuracy
figure;
plot(result.d, result.KNN.AvgAcc, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.simKNN.AvgAcc, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.SVM.AvgAcc, '-s', 'LineWidth', 2);
legend('k-NN', 'sim-k-NN', 'SVM');
xlabel('Dimensionality of the embeded space');
ylabel('Mean Accuracy');
title(['Mean Accuracy using ', drMethod]);

% Sensitivity
figure;
plot(result.d, result.KNN.AvgSen, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.simKNN.AvgSen, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.SVM.AvgSen, '-s', 'LineWidth', 2);
legend('k-NN', 'sim-k-NN', 'SVM');
xlabel('Dimensionality of the embeded space');
ylabel('Mean Sensitivity');
title(['Mean Sensitivity using ', drMethod]);

% Specificity
figure;
plot(result.d, result.KNN.AvgSpec, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.simKNN.AvgSpec, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.SVM.AvgSpec, '-s', 'LineWidth', 2);
legend('k-NN', 'sim-k-NN', 'SVM');
xlabel('Dimensionality of the embeded space');
ylabel('Mean Specificity');
title(['Mean Specificity using ', drMethod]);

% Time
figure;
plot(result.d, result.KNN.AvgTime, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.simKNN.AvgTime, '-s', 'LineWidth', 2);
hold on
plot(result.d, result.SVM.AvgTime, '-s', 'LineWidth', 2);
legend('k-NN', 'sim-k-NN', 'SVM');
xlabel('Dimensionality of the embeded space');
ylabel('Mean elapsed time');
title(['Mean elapsed time using ', drMethod]);

printStr = strcat(' Results (', drMethod, ', Data = ', dataName, ')');
print(strcat('Mean Time', printStr), '-dsvg');

%% Save Results

colheaders = {' ','Accuracy', 'Sensitivity', 'Specificity', 'Time'};

% k-NN
filename = strcat(dataName, '_KNN');

predRes = [result.d(:), result.KNN.AvgAcc(:), ...
    result.KNN.AvgSen(:), result.KNN.AvgSpec(:), result.KNN.AvgTime(:)];

predRes = num2cell(predRes);

inputCell = [colheaders; predRes];

xlswrite(filename, inputCell, drMethod);
xlswrite(filename, result.KNN.AvgCM, strcat('Confusion Matrix ', drMethod)); % Adding confusion matrix

% sim-k-NN
filename = strcat(dataName, '_sim-KNN');

predRes = [result.d(:), result.simKNN.AvgAcc(:), ...
    result.simKNN.AvgSen(:), result.simKNN.AvgSpec(:), result.simKNN.AvgTime(:)];

predRes = num2cell(predRes);

inputCell = [colheaders; predRes];

xlswrite(filename, inputCell, drMethod);
xlswrite(filename, result.simKNN.AvgCM, strcat('Confusion Matrix ', drMethod)); % Adding confusion matrix

% SVM
filename = strcat(dataName, '_SVM');

predRes = [result.d(:), result.SVM.AvgAcc(:), ...
    result.SVM.AvgSen(:), result.SVM.AvgSpec(:), result.SVM.AvgTime(:)];

predRes = num2cell(predRes);

inputCell = [colheaders; predRes];

xlswrite(filename, inputCell, drMethod);
xlswrite(filename, result.SVM.AvgCM, strcat('Confusion Matrix ', drMethod)); % Adding confusion matrix

out = [result.KNN.AvgTime(end); result.simKNN.AvgTime(end); result.SVM.AvgTime(end)];
end