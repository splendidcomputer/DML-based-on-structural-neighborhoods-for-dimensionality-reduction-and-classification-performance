function [trX, trY, tstX, tstY] = MyCVPartition(nFold, X, Y, varargin)
% This function tries to divide the samples of each class equally in each
% fold.

%     p = inputParser();
%     p.addOptional('reduc', 1);
%     p.parse;
%     reductionPerc = p.Results.reduc;

    reductionPerc = varargin{1};

    trX = [];
    trY = [];
    tstX = [];
    tstY = [];

    testRatio = 1 / nFold;
    trainRatio = 1 - testRatio;
    valRatio = 0;



    classNames = unique(Y);
    classNames= classNames(:)';
    nClass = numel(classNames);

    % Find the smallest class
    smallestClassSize = inf;

    for i = classNames

        initClassSize = sum(Y == i);

        if smallestClassSize > initClassSize

            smallestClassSize = initClassSize;

        end

    end
    
    if smallestClassSize > size(X, 1) / 10
        
        smallestClassSize = inf;
        
    end

    empty_class.Size = [];
    empty_class.X = [];

    class = repmat(empty_class, nClass, 1);

    for i = classNames

        if sum(Y == i) > smallestClassSize

            class(i).Size = round(sum(Y == i) * reductionPerc);
            

        else

            class(i).Size = round(sum(Y == i));

        end
        
        tmpX = X(Y == i, :);
        
        sampledInd = ...
            randsample(sum(Y == i), class(i).Size);

        class(i).X = tmpX(sampledInd, :);

        [trainInd, ~, testInd] = ...
            dividerand(class(i).Size, trainRatio, valRatio, testRatio);

        class(i).TrX = class(i).X(trainInd, :);
        class(i).TstX = class(i).X(testInd, :);

        trX = [trX; class(i).TrX];
        trY = [trY; repmat(i, size(class(i).TrX, 1), 1)];

        tstX = [tstX; class(i).TstX];
        tstY = [tstY; repmat(i, size(class(i).TstX, 1), 1)];

    end

end