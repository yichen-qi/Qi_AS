clearvars  -except dlnet averageGrad averageSqGrad;
close all; clc

% addpath('./customLayers');

%% data
% load network
% load dlnet.mat
inputSize = [14 14];
pixelNums = 12*prod(inputSize);
augmenter = imageDataAugmenter('RandRotation',[0 360]);
% load dataset
trainDatasetPath = 'D:\Autoencoder\train';
validDatasetPath = 'D:\Autoencoder\valid';

trainimds = imageDatastore(trainDatasetPath);
trainAugimds = augmentedImageDatastore(2*inputSize,trainimds,'OutputSizeMode','randcrop','DataAugmentation',augmenter);
validimds = imageDatastore(validDatasetPath);
validAugimds = augmentedImageDatastore(2*inputSize,validimds,'OutputSizeMode','randcrop','DataAugmentation',augmenter);

% initialize plot
[ax1,ax2,lineLossTrain,lineLossValid]=initializePlots();
plotFrequency = 5;

%% training parameters
numEpochs = 5;
% trainBatchSize = 16;
% validBatchSize = 16;
trainBatchSize = 64;
validBatchSize = 64;

trainAugimds.MiniBatchSize = trainBatchSize;
validAugimds.MiniBatchSize = validBatchSize;

if ~exist('averageGrad','var') 
    averageGrad = [];
    averageSqGrad = [];
end

numIterations = floor(trainAugimds.NumObservations*numEpochs/trainBatchSize);

learnRate = 1e-4;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
executionEnvironment = "auto";

%% training
iteration = 0;
start = tic;

% Loop over epochs.
for i = 1:numEpochs
    
    % Reset and shuffle datastore.
    reset(trainAugimds);
    trainAugimds = shuffle(trainAugimds);
    
    % Loop over mini-batches.
    while hasdata(trainAugimds)
        iteration = iteration + 1;
                
        % Read mini-batch of data.
        data = read(trainAugimds);
                       
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < trainBatchSize
            continue
        end
        
        % Extract the images from data store into a cell array.
        images = data{:,1};
        
        % Concatenate the images along the 4th dimension.
        Z = cat(4,images{:});
        Z = im2single(Z);
        X = imresize(Z,0.5);
        
        % Convert mini-batch of data to dlarray and specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end

        % Evaluate model gradients.
        [gradients,dlY,loss] = dlfeval(@modelGradients,dlnet,dlX,dlZ,trainBatchSize,pixelNums);

        % Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad] = ...
            adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
        
        % Every plotFequency iterations, plot the training progress.
        if mod(iteration,plotFrequency) == 0 
            
            reset(validAugimds);
            validData = read(validAugimds);
            validImages = validData{:,1};
            
            % Concatenate the images along the 4th dimension.
            VZ = cat(4,validImages{:});
            VZ = im2single(VZ);
            VX = imresize(VZ,0.5);
            
            % Convert mini-batch of data to dlarray and specify the dimension labels
            % 'SSCB' (spatial, spatial, channel, batch).
            dlVX = dlarray(VX, 'SSCB');
            dlVZ = dlarray(VZ, 'SSCB');

            % If training on a GPU, then convert data to gpuArray.
            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                dlVX = gpuArray(dlVX);
                dlVZ = gpuArray(dlVZ);
            end
        
            dlVY = forward(dlnet,dlVX);
            lossValid = sum(abs(dlVY-dlVZ),'all')/(trainBatchSize*pixelNums);
            
            % Use the first image of the mini-batch as a validation image.
            dlVX = dlVX(:,:,:,1);
            dlVY = dlVY(:,:,:,1);
            dlVZ = dlVZ(:,:,:,1);
            
            % To use the function imshow, convert to uint8.
            LRImage = gather(extractdata(dlVX));
            HRImage = gather(extractdata(dlVY));
            GTImage = gather(extractdata(dlVZ));
            
            % Plot the input image and the output image and increase size
            imshow(imtile({LRImage,HRImage,GTImage},'GridSize', [1 3]),'Parent',ax2);
            addpoints(lineLossValid,iteration,double(gather(extractdata(lossValid))))
        end
        
        % Display time elapsed since start of training and training completion percentage.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        completionPercentage = round(iteration/numIterations*100,2);
        title(ax1,"Epoch: " + i + ", Iteration: " + iteration +" of "+ numIterations + "(" + completionPercentage + "%)"+...
            ", LearnRate: "+ learnRate + ", Elapsed: " + string(D))
        drawnow
      
    end
    
    if mod(iteration,2e5) == 0
        learnRate = learnRate*0.5;
    end
    
end

% save('EDSR_trained.mat','dlnet','averageGrad','averageSqGrad');

%% netloss

function [gradients,dlY,loss] = modelGradients(dlnet,dlX,dlZ,trainBatchSize,pixelNums)

    dlY = forward(dlnet,dlX);

    loss = sum(abs(dlY-dlZ),'all')/(trainBatchSize*pixelNums);

    gradients = dlgradient(loss,dlnet.Learnables);

end
