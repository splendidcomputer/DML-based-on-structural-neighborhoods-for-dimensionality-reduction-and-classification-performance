clc;
clear;
close all;
addpath('..\common');
addpath('.\lib');

kf = 10; % number of kfold partitions
kn = 3; % number of nearset neighbor
p =  0; % dimensionality of output, if p set to zero then it will be
% determined automatically

initDataX = xlsread('DataSets\Vehicle\Samples.xlsx');    % Initial data.
initDataY = xlsread('DataSets\Vehicle\Labels.xlsx');

X = initDataX';%[Normal;Dos;Rtol;Utor;Probe]';
[d, N] = size(X);
y = initDataY';
n = N;

c = max(y) - min(y) + 1;    % Number of classes
DLSRAcc = zeros(kf, 1);

for it = 1 : kf
    
    [X,y,XTe,yTe] = kfold(X,y,kf,c);
    N = size(X,2);
    
    ns = 0; nd = 0;
    
    % compute mean of each class
    m = zeros(d,c);
    for i=1:c
        m(:,i) = mean(X(:,y == i),2);
    end
    X = [X,m]; y = [y,(1:c)];
    
    
    % set every point similar to its mean and disimilar to the means of the
    % other classes
    S = zeros(N,2);
    D = zeros((c-1)*N,2);
    for i=1:N
        for j=1:c
            if(j == y(i))
                ns = ns + 1;
                S(ns,1) = i; S(ns,2) = N+j;
            else
                nd = nd+1;
                D(nd,1) = i; D(nd,2) = N+j;
            end
        end
    end
    
    [W,~,p] = learn_mahalanobis_metric(X',S,D,p);
    Z = W' * X;
    ZTe = W' * XTe;
    mdl = fitcknn(Z',y','NumNeighbors',kn);
    classout = predict(mdl, ZTe');
    cp = classperf(yTe, classout);
    display(cp.CorrectRate);
    DLSRAcc(it) = cp.CorrectRate;
    
end

xlswrite('DLSRAcc' ,DLSRAcc);

%% Show results

figure;
plot(DLSRAcc, '-s', 'LineWidth', 2);
hold on
plot(repmat(mean(DLSRAcc), it, 1), '--', 'LineWidth', 2);
legend('Accuracy in each iteration', 'Accuracy on average');
xlabel('Iteration');
ylabel('Accuracy');
