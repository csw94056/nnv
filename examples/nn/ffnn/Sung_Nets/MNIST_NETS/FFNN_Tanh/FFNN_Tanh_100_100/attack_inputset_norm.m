load tanh_100_100_images_normalized.mat;
load MNIST_tanh_100_100_normalized_DenseNet.mat;
N = size(IM_labels, 1); % number of test images used to testing robustness

% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:50                                    % preview first 36 samples
%     subplot(5,10,i)                              % plot them in 6 x 6 grid
%     digit = reshape(IM_data(:,i), [28,28]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(IM_labels(i))                   % show the label
% end

% image perturbation by infinity norm attack (offsetting aimages)
eps = 0.001
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0001(n) = Star(lb, ub);
   A_eps_norm_0001(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0001(n) = RStar(lb, ub, inf);
end

eps = 0.002
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0002(n) = Star(lb, ub);
   A_eps_norm_0002(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0002(n) = RStar(lb, ub, inf);
end

eps = 0.003
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0003(n) = Star(lb, ub);
   A_eps_norm_0003(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0003(n) = RStar(lb, ub, inf);
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

eps = 0.005
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0005(n) = Star(lb, ub);
   A_eps_norm_0005(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0005(n) = RStar(lb, ub, inf);
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

eps = 0.007
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0007(n) = Star(lb, ub);
   A_eps_norm_0007(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0007(n) = RStar(lb, ub, inf);
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

eps = 0.009
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0009(n) = Star(lb, ub);
   A_eps_norm_0009(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0009(n) = RStar(lb, ub, inf);
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

% eps = 0.012
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0012(n) = Star(lb, ub);
%    A_eps_norm_0012(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0012(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.014
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0014(n) = Star(lb, ub);
%    A_eps_norm_0014(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0014(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.016
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0016(n) = Star(lb, ub);
%    A_eps_norm_0016(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0016(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.018
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0018(n) = Star(lb, ub);
%    A_eps_norm_0018(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0018(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.020
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0020(n) = Star(lb, ub);
%    A_eps_norm_0020(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.022
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0022(n) = Star(lb, ub);
%    A_eps_norm_0022(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0022(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.024
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0024(n) = Star(lb, ub);
%    A_eps_norm_0024(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0024(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.026
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0026(n) = Star(lb, ub);
%    A_eps_norm_0026(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0026(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.028
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0028(n) = Star(lb, ub);
%    A_eps_norm_0028(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0028(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.030
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_0030(n) = Star(lb, ub);
%    A_eps_norm_0030(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_0030(n) = RStar(lb, ub, inf);
% end

save tanh_100_100_normalized/inputStar_norm0001.mat S_eps_norm_0001
save tanh_100_100_normalized/inputStar_norm0002.mat S_eps_norm_0002
save tanh_100_100_normalized/inputStar_norm0003.mat S_eps_norm_0003
save tanh_100_100_normalized/inputStar_norm0004.mat S_eps_norm_0004
save tanh_100_100_normalized/inputStar_norm0005.mat S_eps_norm_0005
save tanh_100_100_normalized/inputStar_norm0006.mat S_eps_norm_0006
save tanh_100_100_normalized/inputStar_norm0007.mat S_eps_norm_0007
save tanh_100_100_normalized/inputStar_norm0008.mat S_eps_norm_0008
save tanh_100_100_normalized/inputStar_norm0009.mat S_eps_norm_0009
save tanh_100_100_normalized/inputStar_norm0010.mat S_eps_norm_0010
% save normalized/inputStar_norm0012.mat S_eps_norm_0012
% save normalized/inputStar_norm0014.mat S_eps_norm_0014
% save normalized/inputStar_norm0016.mat S_eps_norm_0016
% save normalized/inputStar_norm0018.mat S_eps_norm_0018
% save normalized/inputStar_norm0020.mat S_eps_norm_0020
% save normalized/inputStar_norm0022.mat S_eps_norm_0022
% save normalized/inputStar_norm0024.mat S_eps_norm_0024
% save normalized/inputStar_norm0026.mat S_eps_norm_0026
% save normalized/inputStar_norm0028.mat S_eps_norm_0028
% save normalized/inputStar_norm0030.mat S_eps_norm_0030

save tanh_100_100_normalized/inputAbsDom_norm0001.mat A_eps_norm_0001
save tanh_100_100_normalized/inputAbsDom_norm0002.mat A_eps_norm_0002
save tanh_100_100_normalized/inputAbsDom_norm0003.mat A_eps_norm_0003
save tanh_100_100_normalized/inputAbsDom_norm0004.mat A_eps_norm_0004
save tanh_100_100_normalized/inputAbsDom_norm0005.mat A_eps_norm_0005
save tanh_100_100_normalized/inputAbsDom_norm0006.mat A_eps_norm_0006
save tanh_100_100_normalized/inputAbsDom_norm0007.mat A_eps_norm_0007
save tanh_100_100_normalized/inputAbsDom_norm0008.mat A_eps_norm_0008
save tanh_100_100_normalized/inputAbsDom_norm0009.mat A_eps_norm_0009
save tanh_100_100_normalized/inputAbsDom_norm0010.mat A_eps_norm_0010
% save normalized/inputAbsDom_norm0012.mat A_eps_norm_0012
% save normalized/inputAbsDom_norm0014.mat A_eps_norm_0014
% save normalized/inputAbsDom_norm0016.mat A_eps_norm_0016
% save normalized/inputAbsDom_norm0018.mat A_eps_norm_0018
% save normalized/inputAbsDom_norm0020.mat A_eps_norm_0020
% save normalized/inputAbsDom_norm0022.mat A_eps_norm_0022
% save normalized/inputAbsDom_norm0024.mat A_eps_norm_0024
% save normalized/inputAbsDom_norm0026.mat A_eps_norm_0026
% save normalized/inputAbsDom_norm0028.mat A_eps_norm_0028
% save normalized/inputAbsDom_norm0030.mat A_eps_norm_0030

save tanh_100_100_normalized/inputRStar_norm0001.mat RS_eps_norm_0001 
save tanh_100_100_normalized/inputRStar_norm0002.mat RS_eps_norm_0002
save tanh_100_100_normalized/inputRStar_norm0003.mat RS_eps_norm_0003
save tanh_100_100_normalized/inputRStar_norm0004.mat RS_eps_norm_0004
save tanh_100_100_normalized/inputRStar_norm0005.mat RS_eps_norm_0005
save tanh_100_100_normalized/inputRStar_norm0006.mat RS_eps_norm_0006
save tanh_100_100_normalized/inputRStar_norm0007.mat RS_eps_norm_0007
save tanh_100_100_normalized/inputRStar_norm0008.mat RS_eps_norm_0008
save tanh_100_100_normalized/inputRStar_norm0009.mat RS_eps_norm_0009
save tanh_100_100_normalized/inputRStar_norm0010.mat RS_eps_norm_0010 
% save normalized/inputRStar_norm0012.mat RS_eps_norm_0012 
% save normalized/inputRStar_norm0014.mat RS_eps_norm_0014 
% save normalized/inputRStar_norm0016.mat RS_eps_norm_0016 
% save normalized/inputRStar_norm0018.mat RS_eps_norm_0018 
% save normalized/inputRStar_norm0020.mat RS_eps_norm_0020 
% save normalized/inputRStar_norm0022.mat RS_eps_norm_0022
% save normalized/inputRStar_norm0024.mat RS_eps_norm_0024
% save normalized/inputRStar_norm0026.mat RS_eps_norm_0026
% save normalized/inputRStar_norm0028.mat RS_eps_norm_0028
% save normalized/inputRStar_norm0030.mat RS_eps_norm_0030




% eps = 0.001
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_001(n) = Star(lb, ub);
%    A_eps_norm_001(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_001(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.002
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_002(n) = Star(lb, ub);
%    A_eps_norm_002(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_002(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.003
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_003(n) = Star(lb, ub);
%    A_eps_norm_003(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_003(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.004
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_004(n) = Star(lb, ub);
%    A_eps_norm_004(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_004(n) = RStar(lb, ub, inf);
% end
% 
% % eps = 0.005
% % for n = 1:N
% %    IM = IM_data(:, n);
% %    lb = IM - eps;
% %    ub = IM + eps;
% %    S_eps_norm_005(n) = Star(lb, ub);
% %    A_eps_norm_005(n) = AbsDom(lb, ub, inf);
% %    RS_eps_norm_005(n) = RStar(lb, ub, inf);
% % end
% 
% eps = 0.006
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_006(n) = Star(lb, ub);
%    A_eps_norm_006(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_006(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.007
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_007(n) = Star(lb, ub);
%    A_eps_norm_007(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_007(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.008
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_008(n) = Star(lb, ub);
%    A_eps_norm_008(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_008(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.009
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_norm_009(n) = Star(lb, ub);
%    A_eps_norm_009(n) = AbsDom(lb, ub, inf);
%    RS_eps_norm_009(n) = RStar(lb, ub, inf);
% end
% 
% % eps = 0.010
% % for n = 1:N
% %    IM = IM_data(:, n);
% %    lb = IM - eps;
% %    ub = IM + eps;
% %    S_eps_norm_010(n) = Star(lb, ub);
% %    A_eps_norm_010(n) = AbsDom(lb, ub, inf);
% %    RS_eps_norm_010(n) = RStar(lb, ub, inf);
% % end
% % 
% % eps = 0.015
% % for n = 1:N
% %    IM = IM_data(:, n);
% %    lb = IM - eps;
% %    ub = IM + eps;
% %    S_eps_norm_015(n) = Star(lb, ub);
% %    A_eps_norm_015(n) = AbsDom(lb, ub, inf);
% %    RS_eps_norm_015(n) = RStar(lb, ub, inf);
% % end
% % 
% % eps = 0.020
% % for n = 1:N
% %    IM = IM_data(:, n);
% %    lb = IM - eps;
% %    ub = IM + eps;
% %    S_eps_norm_020(n) = Star(lb, ub);
% %    A_eps_norm_020(n) = AbsDom(lb, ub, inf);
% %    RS_eps_norm_020(n) = RStar(lb, ub, inf);
% % end
% % 
% % eps = 0.025
% % for n = 1:N
% %    IM = IM_data(:, n);
% %    lb = IM - eps;
% %    ub = IM + eps;
% %    S_eps_norm_025(n) = Star(lb, ub);
% %    A_eps_norm_025(n) = AbsDom(lb, ub, inf);
% %    RS_eps_norm_025(n) = RStar(lb, ub, inf);
% % end
% % 
% % eps = 0.030
% % for n = 1:N
% %    IM = IM_data(:, n);
% %    lb = IM - eps;
% %    ub = IM + eps;
% %    S_eps_norm_030(n) = Star(lb, ub);
% %    A_eps_norm_030(n) = AbsDom(lb, ub, inf);
% %    RS_eps_norm_030(n) = RStar(lb, ub, inf);
% % end
% 
% save normalized/inputStar_norm001.mat S_eps_norm_001
% save normalized/inputStar_norm002.mat S_eps_norm_002 
% save normalized/inputStar_norm003.mat S_eps_norm_003 
% save normalized/inputStar_norm004.mat S_eps_norm_004 
% % save normalized/inputStar_norm005.mat S_eps_norm_005 
% save normalized/inputStar_norm006.mat S_eps_norm_006 
% save normalized/inputStar_norm007.mat S_eps_norm_007 
% save normalized/inputStar_norm008.mat S_eps_norm_008 
% save normalized/inputStar_norm009.mat S_eps_norm_009 
% % save normalized/inputStar_norm010.mat S_eps_norm_010 
% % save normalized/inputStar_norm015.mat S_eps_norm_015 
% % save normalized/inputStar_norm020.mat S_eps_norm_020 
% % save normalized/inputStar_norm025.mat S_eps_norm_025 
% % save normalized/inputStar_norm030.mat S_eps_norm_030
% 
% save normalized/inputAbsDom_norm001.mat A_eps_norm_001
% save normalized/inputAbsDom_norm002.mat A_eps_norm_002 
% save normalized/inputAbsDom_norm003.mat A_eps_norm_003 
% save normalized/inputAbsDom_norm004.mat A_eps_norm_004 
% % save normalized/inputAbsDom_norm005.mat A_eps_norm_005 
% save normalized/inputAbsDom_norm006.mat A_eps_norm_006 
% save normalized/inputAbsDom_norm007.mat A_eps_norm_007 
% save normalized/inputAbsDom_norm008.mat A_eps_norm_008 
% save normalized/inputAbsDom_norm009.mat A_eps_norm_009 
% % save normalized/inputAbsDom_norm010.mat A_eps_norm_010 
% % save normalized/inputAbsDom_norm015.mat A_eps_norm_015 
% % save normalized/inputAbsDom_norm020.mat A_eps_norm_020 
% % save normalized/inputAbsDom_norm025.mat A_eps_norm_025 
% % save normalized/inputAbsDom_norm030.mat A_eps_norm_030
% 
% save normalized/inputRStar_norm001.mat RS_eps_norm_001
% save normalized/inputRStar_norm002.mat RS_eps_norm_002 
% save normalized/inputRStar_norm003.mat RS_eps_norm_003 
% save normalized/inputRStar_norm004.mat RS_eps_norm_004 
% % save normalized/inputRStar_norm005.mat RS_eps_norm_005 
% save normalized/inputRStar_norm006.mat RS_eps_norm_006 
% save normalized/inputRStar_norm007.mat RS_eps_norm_007 
% save normalized/inputRStar_norm008.mat RS_eps_norm_008 
% save normalized/inputRStar_norm009.mat RS_eps_norm_009
% % save normalized/inputRStar_norm010.mat RS_eps_norm_010 
% % save normalized/inputRStar_norm015.mat RS_eps_norm_015 
% % save normalized/inputRStar_norm020.mat RS_eps_norm_020
% % save normalized/inputRStar_norm025.mat RS_eps_norm_025 
% % save normalized/inputRStar_norm030.mat RS_eps_norm_030
