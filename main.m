% In the name of God.
%%
% |Distance metric learning on the nonlinearly dimensionality reduced space.|

clc;
clear;
close all;


%% Load the dataset
%
dataNames = {'Vehicle', 'KDD', 'Bupa', 'Glass', 'Ionosphere', 'Monks', ...
    'New-thyroid', 'Pima', 'WDBC', 'Iris','Wine', 'Wholesale', 'CRC'};

% dataNames = {'CRC'};


res = zeros(3, 1);

for dataName = dataNames
    
    dataName = cell2mat(dataName);
    
    if strcmp(dataName, 'KDD')
        
        initDataX = load('DataSets\KDD\kddData.mat');    % Initial data.
        initDataX = initDataX.initTrX;
        initDataY = load('DataSets\KDD\kddLabel.mat');
        initDataY = initDataY.trY;
        
    else
        
        initDataX = xlsread(strcat('DataSets\', dataName, '\Samples.xlsx'));    % Initial data.
        initDataY = xlsread(strcat('DataSets\', dataName, '\Labels.xlsx'));
        
    end
    
    %     initDataX = load('DataSets\20news\train.data');    % Initial data.
    %     initDataY = load('DataSets\20news\train.label');
    
            drMethods = {'PCA', 'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', ...
                'Autoencoder'};
    
%     drMethods = {'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', ...
%         'Autoencoder'};
    
    
    for drMethod = drMethods
        
        drMethod = cell2mat(drMethod);
        
        res = [res; MyDML(dataName, initDataX, initDataY, drMethod)];
        %         MyDMLOld(dataName, initDataX, initDataY, drMethod);
        
    end
    
end

filename = 'Results';
xlswrite(filename, res);