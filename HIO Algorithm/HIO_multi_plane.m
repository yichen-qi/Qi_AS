clearvars;
close all;clc
%%
% load test_holo_com.mat
%This file contains all the information for image pre-processing, which I am not authorized to upload.
load com_facet.mat 
test_holo = AmpProx;
% load calibration_phase.mat
figure,imagesc(abs(test_holo));%farbig Bild
figure,imshow(abs(test_holo));
% axis image off

[Ny,Nx] = size(test_holo);

dp = 2.2e-6; %pixel
lambda = 532e-9;
z = 0.0736; %0.0807;
zs = [0.06 0.0736 0.08];
imgNum = length(zs);

A = [];
for k = 1:imgNum
    [Uk,~] = prop(test_holo,dp,dp,lambda,zs(k)-z);
    Ak = abs(Uk);
    A = cat(3,A,Ak);
end

[U0,phase_raw] = prop(test_holo,dp,dp,lambda,-z);


axis image off
figure,imshow(abs(U0));
figure,imagesc(abs(U0));
mask = imbinarize(abs(U0),0.02);
mask = imclose(mask,strel('disk',25));
figure,imshow(mask);

iters = 2000;
b = 0.9;
% init_phase = 2*pi*rand(size(test_holo));
init_phase = 0;

Uo = mask.*exp(1i*init_phase);%?

figure,h = animatedline;
% ylim([0 0.001])
figure,im = imagesc(abs(Uo));
axis image off
t = title('Iteration = 0');
Un = zeros(Ny,Nx,imgNum);
k = 1;
phase_error = [];

%% start iteration

for k = k + 1:k + iters
    
    for n = 1:imgNum
        Ui = prop(Uo,dp,dp,lambda,zs(n));
        Ua = A(:,:,n).*Ui./abs(Ui);
        Um = prop(Ua,dp,dp,lambda,-zs(n));
        Uo = ((1+b)*Um - Uo).*mask + Uo - b*Um;
        Un(:,:,n) = Uo;
    end
    Uo = mean(Un,3);
    
    phase_recon = angle(Uo);
    phase_diff = wrapToPi(phase_recon - phase_raw);
    phase_err_line = phase_diff(:);
    phase_err_line(~mask(:)) = [];
    phase_error(k-1) = circ_std(phase_err_line);
           
    figure(2314);
    subplot(1,2,1);imagesc(phase_diff);axis image;axis off;title('phase difference');
    subplot(1,2,2);plot(phase_error);title('Phase Error');
    pause(0.02);
    
end

figure,imshowpair(abs(Uo),abs(U0),'montage');
