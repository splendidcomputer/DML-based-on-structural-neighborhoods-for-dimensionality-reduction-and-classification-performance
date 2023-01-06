% CORRESPONDENCE INFORMATION
%    This code is written by Feiping Nie
% 
%    Address:    FIT Building 3-120 Room
%                     Tsinghua Univeristy, Beijing, China, 100084
%    Email to:    feipingnie@gmail.com  OR  smxiang@gmail.com, 

%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested using matlab 6.5  and matlab    7.0

%  For more detials, please see the paper:
%  Shiming Xiang, Feiping Nie, and Changshui Zhang. 
%  Learning a Mahalanobis Distance Metric for Data Clustering and Classification. Pattern Recognition, 
%  Volume 41, Issue 12, Pages 3600-3612, 2008.

% %solving the constrained optimization problem
function [W, lamda]  = learn_iteratively(sb, sw, feature_num, p_err)

[evec eval] = eig(sw);
eval = abs(diag(eval));
% nzero = length(find(eval<=1e-6));
d = size(sw,1);
nzero = d - rank(sw,1e-6);
if feature_num <= nzero
    [dumb, iEvals] = sort(eval);
    Z = evec(:,iEvals(1:nzero));
    [evec eval] = eig(Z'*sb*Z);
    [dumb, iEvals] = sort(diag(eval));
    iEvals = iEvals(end:-1:1);
    W = Z*evec(:,iEvals(1:feature_num));
else
    [evec eval] = eig(sb);
    a = sort(diag(eval));
    a = a(end:-1:1);
    a = sum(a(1:feature_num));
    [evec eval] = eig(sw);
    b = sort(diag(eval));
    b = sum(b(1:feature_num));
    lamda1 = a/b;
    lamda2 = trace(sb)/trace(sw);
    interval = lamda1 - lamda2;
    lamda = (lamda1+lamda2)/2;

    while interval > p_err
        [evec eval] = eig(sb - lamda*sw);
        [eval,index] = sort(diag(eval));
        eval = eval(end:-1:1);
        sum_eval = sum(eval(1:feature_num));
        if sum_eval > 0
            lamda2 = lamda;
        else
            lamda1 = lamda;
        end;
        interval = lamda1 - lamda2;
        lamda = (lamda1+lamda2)/2;
    end;
    index = index(end:-1:1);
    W = evec(:,index(1:feature_num));
end;
