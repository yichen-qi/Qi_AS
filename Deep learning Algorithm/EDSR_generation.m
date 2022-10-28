% clearvars -except dlnet;
close all;clc

% addpath('./customLayers')

%% Generate net
depth = 8;
% numFilters = 256;
numFilters = 28;
% inputSize = [256 256];
inputSize = [28 28];

lgraph = layerGraph;
layers = [
%     imageInputLayer([inputSize 3],'Normalization','none','Name','input')
    imageInputLayer([inputSize 1],'Normalization','none','Name','input')
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name','resConvIn')
    ];

lgraph = addLayers(lgraph,layers);

[lgraph,outputName] = resNet(lgraph,'resConvIn',numFilters,1);
for k = 2:depth
[lgraph,outputName] = resNet(lgraph,outputName,numFilters,k);
end

layers = [
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name','resConvOut')
    additionLayer(2,'Name','skipAdd')
    convolution2dLayer([3 3],48,'Stride',[1 1],'Padding','same','Name','shuffleConv')
%     pixelShuffleLayer('pixelShuffle',2)
    convolution2dLayer([1 1],3,'Stride',[1 1],'Padding','same','Name','convOut')
    ];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,'resConvOut');
lgraph = connectLayers(lgraph,'resConvIn','skipAdd/in2');

dlnet = dlnetwork(lgraph);
% save('dlnet.mat','dlnet');

%% denseNet
function [newlgraph,outputName] = resNet(lgraph, inputName, numFilters, order)

k = num2str(order);

Layers = [    
    convolution2dLayer([3,3],numFilters,'Padding','same','Name',['conv1',k])
    reluLayer('Name',['relu',k])
    convolution2dLayer([3,3],numFilters,'Padding','same','Name',['conv2',k])
    scalingLayer('Scale',0.1,'Name',['scaling',k])
    additionLayer(2,'Name',['add',k])
    ];
    
lgraph = addLayers(lgraph,Layers);
lgraph = connectLayers(lgraph,inputName,['conv1',k]);
newlgraph = connectLayers(lgraph,inputName,['add',k,'/in2']);

outputName = ['add',k];

end