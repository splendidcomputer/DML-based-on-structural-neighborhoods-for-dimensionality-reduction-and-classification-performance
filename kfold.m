function [XTr,yTr,XTe,yTe] = kfold(X,y,k,c)
    % c: number of classes
    if(nargin < 4)
        c = size(unique(y),2);
    end
    
    for j=1:c
        class{j} = X(:, y == j);
    end
    
    XTe = []; yTe = [];
    XTr = []; yTr = [];
    
    for j=1:c
        ind = crossvalind('Kfold',size(class{j},2),k);
        nTr = sum(ind ~= k);
        nTe = size(class{j},2) - nTr;
        XTr = [XTr,class{j}(:,ind ~= k)];
        XTe = [XTe,class{j}(:,ind == k)];
        yTr = [yTr, j*ones(1,nTr)];
        yTe = [yTe, j*ones(1,nTe)];
    end
end