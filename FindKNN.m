function nnInd = FindKNN(test, trData, k)

    nTr = size(trData, 1);
    distVec = zeros(1, nTr);
    
    p = 1; % Norm Indicator.
    
    for i = 1 : nTr
       
        distVec(i) = norm(test - trData(i, :), p);
        
    end
    
    [~, Ind] = sort(distVec);
    
    nnInd = Ind(1 : k);

end