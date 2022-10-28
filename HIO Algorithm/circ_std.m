function [s s0] = circ_std(alpha, w, d, dim)
% s = circ_std(alpha, w, d, dim)
%   Computes circular standard deviation for circular data 
%   (equ. 26.20, Zar).   计算圆形标准偏差
%
%   Input:
%     alpha	sample of angles in radians 弧度
%     [w		weightings in case of binned angle data]合并角度数据时的权重
%     [d    spacing of bin centers for binned data, if supplied 
%           correction factor is used to correct for bias in 
%           estimation of r]分箱数据的分箱中心间距，如果提供的校正因子用于校正 r 估计中的偏差
%     [dim  compute along this dimension, default is 1]沿此维度计算，默认为 1
%
%     If dim argument is specified, all other optional arguments can be
%     left empty: circ_std(alpha, [], [], dim) 如果指定了 dim 参数，则所有其他可选参数可以留空
%
%   Output:
%     s     angular deviation  角度偏差
%     s0    circular standard deviation 圆形标准差
%
% PHB 6/7/2008
%
% References:
%   Biostatistical Analysis, J. H. Zar
%
% Circular Statistics Toolbox for Matlab

% By Philipp Berens, 2009
% berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

if nargin < 4
  dim = 1;
end

if nargin < 3 || isempty(d)
  % per default do not apply correct for binned data
  d = 0;
end

if nargin < 2 || isempty(w)
  % if no specific weighting has been specified
  % assume no binning has taken place
	w = ones(size(alpha));
else
  if size(w,2) ~= size(alpha,2) || size(w,1) ~= size(alpha,1) 
    error('Input dimensions do not match');
  end 
end

% compute mean resultant vector length
r = circ_r(alpha,w,d,dim);

s = sqrt(2*(1-r));      % 26.20
s0 = sqrt(-2*log(r));    % 26.21



