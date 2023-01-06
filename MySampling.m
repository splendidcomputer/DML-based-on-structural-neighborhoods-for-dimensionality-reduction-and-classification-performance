function [sampledX, sampledY] = MySampling(initTrX, trY)

    maxSampleSize = 1000;
    
    if size(initTrX, 1) < maxSampleSize

        sampledX = initTrX;
        sampledY = trY;
        return;
        
    end
    
    sampledX = [];
    sampledY = [];
    
    classNames = unique(trY);
    classNames = classNames(:)';
    
    nClass = numel(classNames);
    
    % Create class prototypes
    emty_class.Size = [];
    empty_class.X = [];
    
    classes = repmat(empty_class, nClass, 1);
    
    % Find the smallest class
    smallestClassSize = inf;
    
    for i = classNames
       
        initClassSize = sum(trY == i);
        
        if smallestClassSize > initClassSize
            
            smallestClassSize = initClassSize;
            
        end
        
    end
    
    for i = classNames
       
        initClassSize = sum(trY == i);
        tmpX = initTrX(trY == i, :);
        class(i).X = tmpX(randsample(initClassSize, smallestClassSize), :);
        
        sampledX = [sampledX; class(i).X];
        sampledY = [sampledY; repmat(i, size(class(i).X, 1), 1)];
        
    end
    

end