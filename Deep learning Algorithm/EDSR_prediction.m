%% Prediction
load dlnet.mat
X = im2single(imread('C:\Users\r5000\Desktop\Deep learning\2.png'));
X = rgb2gray(X);
dlX = dlarray(X,'SSCB');
dlX = gpuArray(dlX);

dlY = forward(dlnet,dlX);
Y = gather(extractdata(dlY));
Y = rgb2gray(Y);
% figure,imshowpair(imresize(X,2),Y,'montage');

figure,imshowpair(imresize(X,1),Y,'montage');