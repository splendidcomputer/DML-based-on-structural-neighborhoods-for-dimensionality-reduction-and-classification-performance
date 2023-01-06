function testResults = MultiSVM(trX, trY, tstX)

    testResults = zeros(size(tstX, 1), 1);
    classNames = unique(trY);
    classNames = classNames(:)';
    nClass = numel(classNames);
    SVMModels = cell(nClass, 1);
    

    rng(1); % For reproducibility

    for i = classNames

        tmpLabels = ones(size(trY));
        tmpLabels(trY ~= i) = 0;
        SVMModels{i} = fitcsvm(trX, tmpLabels, 'ClassNames',[false true],'Standardize',false,...
            'KernelFunction','rbf','BoxConstraint',1);
        [A{i}, B{i}] = predict(SVMModels{i},tstX);
        
        tmp = logical([A{i}]);

        testResults(tmp) = i;
        
    end
    
    postiveScores = cell2mat(B);
    postiveScores(:, 1 : 2 : end) = [];
    [~, highestScoreInd] = max(postiveScores, [], 2);
    testResults(testResults == 0) = highestScoreInd(testResults == 0);
    testResults = highestScoreInd;

end