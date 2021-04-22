load images_normalized.mat;
load MNIST_sigmoid_100_100_normalized_DenseNet.mat;
N = size(IM_labels, 1); % number of test images used to testing robustness

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 36 samples
    subplot(5,10,i)                              % plot them in 6 x 6 grid
    digit = reshape(IM_data(:,i), [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(IM_labels(i))                   % show the label
end


% image perturbation by infinity norm attack (offsetting aimages)
eps = 0.002
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0002(n) = Star(lb, ub);
   A_eps_norm_0002(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0002(n) = RStar(lb, ub, inf);
end

eps = 0.004
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0004(n) = Star(lb, ub);
   A_eps_norm_0004(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0004(n) = RStar(lb, ub, inf);
end

eps = 0.006
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0006(n) = Star(lb, ub);
   A_eps_norm_0006(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0006(n) = RStar(lb, ub, inf);
end

eps = 0.008
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0008(n) = Star(lb, ub);
   A_eps_norm_0008(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0008(n) = RStar(lb, ub, inf);
end

eps = 0.010
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0010(n) = Star(lb, ub);
   A_eps_norm_0010(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0010(n) = RStar(lb, ub, inf);
end

eps = 0.012
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0012(n) = Star(lb, ub);
   A_eps_norm_0012(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0012(n) = RStar(lb, ub, inf);
end

eps = 0.014
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0014(n) = Star(lb, ub);
   A_eps_norm_0014(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0014(n) = RStar(lb, ub, inf);
end

eps = 0.016
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0016(n) = Star(lb, ub);
   A_eps_norm_0016(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0016(n) = RStar(lb, ub, inf);
end

eps = 0.018
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0018(n) = Star(lb, ub);
   A_eps_norm_0018(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0018(n) = RStar(lb, ub, inf);
end

eps = 0.020
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0020(n) = Star(lb, ub);
   A_eps_norm_0020(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0020(n) = RStar(lb, ub, inf);
end

eps = 0.022
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0022(n) = Star(lb, ub);
   A_eps_norm_0022(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0022(n) = RStar(lb, ub, inf);
end

eps = 0.024
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0024(n) = Star(lb, ub);
   A_eps_norm_0024(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0024(n) = RStar(lb, ub, inf);
end

eps = 0.026
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0026(n) = Star(lb, ub);
   A_eps_norm_0026(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0026(n) = RStar(lb, ub, inf);
end

eps = 0.028
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0028(n) = Star(lb, ub);
   A_eps_norm_0028(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0028(n) = RStar(lb, ub, inf);
end

eps = 0.030
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0030(n) = Star(lb, ub);
   A_eps_norm_0030(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0030(n) = RStar(lb, ub, inf);
end

save normalized/inputStar_norm0002.mat S_eps_norm_0002
save normalized/inputStar_norm0004.mat S_eps_norm_0004
save normalized/inputStar_norm0006.mat S_eps_norm_0006
save normalized/inputStar_norm0008.mat S_eps_norm_0008
save normalized/inputStar_norm0010.mat S_eps_norm_0010
save normalized/inputStar_norm0012.mat S_eps_norm_0012
save normalized/inputStar_norm0014.mat S_eps_norm_0014
save normalized/inputStar_norm0016.mat S_eps_norm_0016
save normalized/inputStar_norm0018.mat S_eps_norm_0018
save normalized/inputStar_norm0020.mat S_eps_norm_0020
save normalized/inputStar_norm0022.mat S_eps_norm_0022
save normalized/inputStar_norm0024.mat S_eps_norm_0024
save normalized/inputStar_norm0026.mat S_eps_norm_0026
save normalized/inputStar_norm0028.mat S_eps_norm_0028
save normalized/inputStar_norm0030.mat S_eps_norm_0030

save normalized/inputAbsDom_norm0002.mat A_eps_norm_0002
save normalized/inputAbsDom_norm0004.mat A_eps_norm_0004
save normalized/inputAbsDom_norm0006.mat A_eps_norm_0006
save normalized/inputAbsDom_norm0008.mat A_eps_norm_0008
save normalized/inputAbsDom_norm0010.mat A_eps_norm_0010
save normalized/inputAbsDom_norm0012.mat A_eps_norm_0012
save normalized/inputAbsDom_norm0014.mat A_eps_norm_0014
save normalized/inputAbsDom_norm0016.mat A_eps_norm_0016
save normalized/inputAbsDom_norm0018.mat A_eps_norm_0018
save normalized/inputAbsDom_norm0020.mat A_eps_norm_0020
save normalized/inputAbsDom_norm0022.mat A_eps_norm_0022
save normalized/inputAbsDom_norm0024.mat A_eps_norm_0024
save normalized/inputAbsDom_norm0026.mat A_eps_norm_0026
save normalized/inputAbsDom_norm0028.mat A_eps_norm_0028
save normalized/inputAbsDom_norm0030.mat A_eps_norm_0030

save normalized/inputRStar_norm0002.mat RS_eps_norm_0002 
save normalized/inputRStar_norm0004.mat RS_eps_norm_0004 
save normalized/inputRStar_norm0006.mat RS_eps_norm_0006 
save normalized/inputRStar_norm0008.mat RS_eps_norm_0008 
save normalized/inputRStar_norm0010.mat RS_eps_norm_0010 
save normalized/inputRStar_norm0012.mat RS_eps_norm_0012 
save normalized/inputRStar_norm0014.mat RS_eps_norm_0014 
save normalized/inputRStar_norm0016.mat RS_eps_norm_0016 
save normalized/inputRStar_norm0018.mat RS_eps_norm_0018 
save normalized/inputRStar_norm0020.mat RS_eps_norm_0020 
save normalized/inputRStar_norm0022.mat RS_eps_norm_0022
save normalized/inputRStar_norm0024.mat RS_eps_norm_0024
save normalized/inputRStar_norm0026.mat RS_eps_norm_0026
save normalized/inputRStar_norm0028.mat RS_eps_norm_0028
save normalized/inputRStar_norm0030.mat RS_eps_norm_0030


% eps = 0.02
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_002(n) = Star(lb, ub);
%    A_eps_norm_002(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_002(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.04
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_004(n) = Star(lb, ub);
%    A_eps_norm_004(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_004(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.06
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_006(n) = Star(lb, ub);
%    A_eps_norm_006(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_006(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.08
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_008(n) = Star(lb, ub);
%    A_eps_norm_008(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_008(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.10
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_010(n) = Star(lb, ub);
%    A_eps_norm_010(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_010(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.12
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_012(n) = Star(lb, ub);
%    A_eps_norm_012(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_012(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.14
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_014(n) = Star(lb, ub);
%    A_eps_norm_014(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_014(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.16
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_016(n) = Star(lb, ub);
%    A_eps_norm_016(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_016(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.18
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_018(n) = Star(lb, ub);
%    A_eps_norm_018(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_018(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.20
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_020(n) = Star(lb, ub);
%    A_eps_norm_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.22
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_020(n) = Star(lb, ub);
%    A_eps_norm_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.24
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_020(n) = Star(lb, ub);
%    A_eps_norm_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.26
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_020(n) = Star(lb, ub);
%    A_eps_norm_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_020(n) = RStar(lb, ub, inf);
% end
% 
% save normalized/inputStar_norm002.mat S_eps_norm_002
% save normalized/inputStar_norm004.mat S_eps_norm_004
% save normalized/inputStar_norm006.mat S_eps_norm_006 
% save normalized/inputStar_norm008.mat S_eps_norm_008 
% save normalized/inputStar_norm010.mat S_eps_norm_010
% save normalized/inputStar_norm012.mat S_eps_norm_012 
% save normalized/inputStar_norm014.mat S_eps_norm_014 
% save normalized/inputStar_norm016.mat S_eps_norm_016 
% save normalized/inputStar_norm018.mat S_eps_norm_018 
% save normalized/inputStar_norm020.mat S_eps_norm_020 
% save normalized/inputStar_norm020.mat S_eps_norm_022 
% save normalized/inputStar_norm020.mat S_eps_norm_024 
% save normalized/inputStar_norm020.mat S_eps_norm_026 
% 
% save normalized/inputAbsDom_norm002.mat A_eps_norm_002
% save normalized/inputAbsDom_norm004.mat A_eps_norm_004
% save normalized/inputAbsDom_norm006.mat A_eps_norm_006
% save normalized/inputAbsDom_norm008.mat A_eps_norm_008
% save normalized/inputAbsDom_norm010.mat A_eps_norm_010
% save normalized/inputAbsDom_norm012.mat A_eps_norm_012
% save normalized/inputAbsDom_norm014.mat A_eps_norm_014
% save normalized/inputAbsDom_norm016.mat A_eps_norm_016
% save normalized/inputAbsDom_norm018.mat A_eps_norm_018
% save normalized/inputAbsDom_norm020.mat A_eps_norm_020
% save normalized/inputAbsDom_norm020.mat A_eps_norm_022
% save normalized/inputAbsDom_norm020.mat A_eps_norm_024
% save normalized/inputAbsDom_norm020.mat A_eps_norm_026
% 
% save normalized/inputRStar_norm002.mat RS_eps_norm_002 
% save normalized/inputRStar_norm004.mat RS_eps_norm_004 
% save normalized/inputRStar_norm006.mat RS_eps_norm_006 
% save normalized/inputRStar_norm008.mat RS_eps_norm_008 
% save normalized/inputRStar_norm010.mat RS_eps_norm_010 
% save normalized/inputRStar_norm012.mat RS_eps_norm_012 
% save normalized/inputRStar_norm014.mat RS_eps_norm_014 
% save normalized/inputRStar_norm016.mat RS_eps_norm_016 
% save normalized/inputRStar_norm018.mat RS_eps_norm_018 
% save normalized/inputRStar_norm020.mat RS_eps_norm_020 
% save normalized/inputRStar_norm020.mat RS_eps_norm_022 
% save normalized/inputRStar_norm020.mat RS_eps_norm_024 
% save normalized/inputRStar_norm020.mat RS_eps_norm_026 

% eps = 0.2
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_02(n) = Star(lb, ub);
%    A_eps_norm_02(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_02(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.4
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_04(n) = Star(lb, ub);
%    A_eps_norm_04(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_04(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.6
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_06(n) = Star(lb, ub);
%    A_eps_norm_06(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_06(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.8
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_08(n) = Star(lb, ub);
%    A_eps_norm_08(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_08(n) = RStar(lb, ub, inf);
% end
% 
% eps = 1.0
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_10(n) = Star(lb, ub);
%    A_eps_norm_10(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_10(n) = RStar(lb, ub, inf);
% end
% 
% eps = 1.2
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_12(n) = Star(lb, ub);
%    A_eps_norm_12(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_12(n) = RStar(lb, ub, inf);
% end
% 
% eps = 1.4
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_14(n) = Star(lb, ub);
%    A_eps_norm_14(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_14(n) = RStar(lb, ub, inf);
% end
% 
% eps = 1.6
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_16(n) = Star(lb, ub);
%    A_eps_norm_16(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_16(n) = RStar(lb, ub, inf);
% end
% 
% eps = 1.8
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_18(n) = Star(lb, ub);
%    A_eps_norm_18(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_18(n) = RStar(lb, ub, inf);
% end
% 
% eps = 2.0
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_20(n) = Star(lb, ub);
%    A_eps_norm_20(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_20(n) = RStar(lb, ub, inf);
% end

% save normalized/inputStar_norm02.mat S_eps_norm_02
% save normalized/inputStar_norm04.mat S_eps_norm_04 
% save normalized/inputStar_norm06.mat S_eps_norm_06 
% save normalized/inputStar_norm08.mat S_eps_norm_08 
% save normalized/inputStar_norm10.mat S_eps_norm_10
% save normalized/inputStar_norm12.mat S_eps_norm_12 
% save normalized/inputStar_norm14.mat S_eps_norm_14 
% save normalized/inputStar_norm16.mat S_eps_norm_16 
% save normalized/inputStar_norm18.mat S_eps_norm_18 
% save normalized/inputStar_norm20.mat S_eps_norm_20 
% 
% save normalized/inputAbsDom_norm02.mat A_eps_norm_02
% save normalized/inputAbsDom_norm04.mat A_eps_norm_04
% save normalized/inputAbsDom_norm06.mat A_eps_norm_06
% save normalized/inputAbsDom_norm08.mat A_eps_norm_08
% save normalized/inputAbsDom_norm10.mat A_eps_norm_10
% save normalized/inputAbsDom_norm12.mat A_eps_norm_12
% save normalized/inputAbsDom_norm14.mat A_eps_norm_14
% save normalized/inputAbsDom_norm16.mat A_eps_norm_16
% save normalized/inputAbsDom_norm18.mat A_eps_norm_18
% save normalized/inputAbsDom_norm20.mat A_eps_norm_20
% 
% save normalized/inputRStar_norm02.mat RS_eps_norm_02 
% save normalized/inputRStar_norm04.mat RS_eps_norm_04 
% save normalized/inputRStar_norm06.mat RS_eps_norm_06 
% save normalized/inputRStar_norm08.mat RS_eps_norm_08 
% save normalized/inputRStar_norm10.mat RS_eps_norm_10 
% save normalized/inputRStar_norm12.mat RS_eps_norm_12 
% save normalized/inputRStar_norm14.mat RS_eps_norm_14 
% save normalized/inputRStar_norm16.mat RS_eps_norm_16 
% save normalized/inputRStar_norm18.mat RS_eps_norm_18 
% save normalized/inputRStar_norm20.mat RS_eps_norm_20 

% eps = 0.005;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_005(n) = Star(lb, ub);
%    A_eps_norm_005(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_005(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.010;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_010(n) = Star(lb, ub);
%    A_eps_norm_010(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_010(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.015;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_015(n) = Star(lb, ub);
%    A_eps_norm_015(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_015(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.020;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_020(n) = Star(lb, ub);
%    A_eps_norm_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.025;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_025(n) = Star(lb, ub);
%    A_eps_norm_025(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_025(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.030;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_030(n) = Star(lb, ub);
%    A_eps_norm_030(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_030(n) = RStar(lb, ub, inf);
% end
%  
% save normalized/inputStar_norm005.mat S_eps_norm_005 
% save normalized/inputStar_norm010.mat S_eps_norm_010 
% save normalized/inputStar_norm015.mat S_eps_norm_015 
% save normalized/inputStar_norm020.mat S_eps_norm_020 
% save normalized/inputStar_norm025.mat S_eps_norm_025 
% save normalized/inputStar_norm030.mat S_eps_norm_030
% 
% save normalized/inputAbsDom_norm005.mat A_eps_norm_005 
% save normalized/inputAbsDom_norm010.mat A_eps_norm_010 
% save normalized/inputAbsDom_norm015.mat A_eps_norm_015 
% save normalized/inputAbsDom_norm020.mat A_eps_norm_020 
% save normalized/inputAbsDom_norm025.mat A_eps_norm_025 
% save normalized/inputAbsDom_norm030.mat A_eps_norm_030
% 
% save normalized/inputRStar_norm005.mat RS_eps_norm_005 
% save normalized/inputRStar_norm010.mat RS_eps_norm_010 
% save normalized/inputRStar_norm015.mat RS_eps_norm_015 
% save normalized/inputRStar_norm020.mat RS_eps_norm_020
% save normalized/inputRStar_norm025.mat RS_eps_norm_025 
% save normalized/inputRStar_norm030.mat RS_eps_norm_030
