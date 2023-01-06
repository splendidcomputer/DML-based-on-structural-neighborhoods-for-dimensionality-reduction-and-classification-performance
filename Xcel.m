clc;
clear;
close all;

res = xlsread('Results');

reshaped = reshape(res, 3, numel(res)/3);

KNN = reshaped(1, :);
rKNN = reshape(KNN, 7, 12);
rKNN = rKNN';

KNNpS = reshaped(2, :);
rKNNpS = reshape(KNNpS, 7, 12);
rKNNpS = rKNNpS';


SVM = reshaped(3, :);
rSVM = reshape(SVM, 7, 12);
rSVM = rSVM';

results = [rKNN, rKNNpS, rSVM];
