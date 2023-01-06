function distMat = CalcDist(X)

    p = 2;  % Norm Indicator
    nSamples = size(X, 1);
    distMat = zeros(nSamples);
    
    
    for i = 1 : nSamples - 1
        
        for j = i + 1 : nSamples
            
            distMat(i, j) = norm(X(i, :) - X(j, :), p);
            distMat(j, i) = distMat(i, j);
            
        end
        
    end

end