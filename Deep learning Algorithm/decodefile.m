function output=decodefile(filename,type)
%数据介绍如下，参考网址http://yann.lecun.com/exdb/mnist/index.html

% TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
% 
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
% 0004     32 bit integer  60000            number of items 
% 0008     unsigned byte   ??               label 
% 0009     unsigned byte   ??               label 
% ........ 
% xxxx     unsigned byte   ??               label
% The labels values are 0 to 9.

% TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
% 
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000803(2051) magic number 
% 0004     32 bit integer  60000            number of images 
% 0008     32 bit integer  28               number of rows 
% 0012     32 bit integer  28               number of columns 
% 0016     unsigned byte   ??               pixel 
% 0017     unsigned byte   ??               pixel 
% ........ 
% xxxx     unsigned byte   ??               pixel

% TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
% 
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
% 0004     32 bit integer  10000            number of items 
% 0008     unsigned byte   ??               label 
% 0009     unsigned byte   ??               label 
% ........ 
% xxxx     unsigned byte   ??               label
% The labels values are 0 to 9.
% 
% TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
% 
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000803(2051) magic number 
% 0004     32 bit integer  10000            number of images 
% 0008     32 bit integer  28               number of rows 
% 0012     32 bit integer  28               number of columns 
% 0016     unsigned byte   ??               pixel 
% 0017     unsigned byte   ??               pixel 
% ........ 
% xxxx     unsigned byte   ??               pixel
fio=fopen(filename,'r');%原始文件中数据是以2进制存储的。
a = fread(fio,'uint8');%以8进制方式读取源文件。虽然前几项是32bit的，但是图像像素数据是8bit的，所以此处用8bit处理。

if strcmp(type,'image')
%     magic_num=a(1)*256^3+a(2)*256^2+a(3)*256+a(4);
%     image_num=a(5)*256^3+a(6)*256^2+a(7)*256+a(8);
%     image_rows=a(9)*256^3+a(10)*256^2+a(11)*256+a(12);%默认训练和测试图像都已经reshape到一个size
%     image_cols=a(13)*256^3+a(14)*256^2+a(15)*256+a(16);

    output=a(17:end);%提取像素数据
else if strcmp(type,'label')
%         magic_num=a(1)*256^3+a(2)*256^2+a(3)*256+a(4);
%         image_num=a(5)*256^3+a(6)*256^2+a(7)*256+a(8);
        output=a(9:end);      
    end
end
