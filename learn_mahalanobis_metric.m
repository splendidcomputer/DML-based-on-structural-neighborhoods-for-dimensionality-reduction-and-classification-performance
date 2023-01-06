% CORRESPONDENCE INFORMATION
%    This code is written by Feiping Nie
% 
%    Address:    National Laboratory of Pattern Recognition, 
%    Institute of Automation, Academy of Sciences, Beijing 100190

%    Email to:    feipingnie@gmail.com  OR  smxiang@gmail.com, 

%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS), volumn 23, issue 11, pages 1738-1754, 2012.


%==========================================================================

function [W, A, p] = learn_mahalanobis_metric(X, S, D, p)
% input
% X:         each row is a data point (the source data, or null-space reduced data of source data)
% M:         Similar Set  
% C:         Disimilar Set  
% p:         dimensionality of output

% return:
% W:   the projetcion  matrix
% A:    the Mahalanobis distance metrix 

% How to use after we learned a W or A:
% Method 1----- via W
% for a data point,  namely, a column vector x:   we have y =  W'  * x;
% then we can use K-NN classifier or other classifiers,  clustering  algorithms, and so on

% Method 2----- directly via A
% for a pair of data points x1 and x2 (column vectors),   
% we can get  distance = (x1-x2)' * A * (x1-x2).  That is, we can use matrix A to replace the identity matrix I used in standard Euclidean distance metric
% Thus, this can be used for the tasks of classification, ranking,  clustering, and so on.


    % calculate the Sb and Sw
    X = double(X);
    d = size(X,2);
    Ns = size(S,1);
    Nd = size(D,1);
    Sw = zeros(d,d);
    Sb = zeros(d,d);
    for i=1:Ns
          xi = X(S(i,1),:)';
          xj = X(S(i,2),:)';
          Sw = Sw + (xi-xj)*(xi-xj)';
    end

    for i=1:Nd
          xi = X(D(i,1),:)';
          xj = X(D(i,2),:)';
          Sb = Sb + (xi-xj)*(xi-xj)';
    end
    if(nargin < 4 || p ==0)
        p = d - rank(Sw,1e-6) + 1;
    end
    % load(['Sbw',num2str(j)], 'Sb','Sw');
    p_error = 0.00001;
    W =  learn_iteratively(Sb, Sw, p, p_error);

    % Thus, we get the matrix of the Mahalanobis distance metrix 
    A = W * W';

return;