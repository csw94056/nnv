load tanh_100_100_images.mat;
load MNIST_tanh_100_100_DenseNet.mat;
N = size(IM_labels, 1); % number of test images used to testing robustness

% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:50                                    % preview first 36 samples
%     subplot(5,10,i)                              % plot them in 6 x 6 grid
%     digit = reshape(IM_data(:,i), [28,28]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(IM_labels(i))                   % show the label
% end


% image perturbation by infinity norm attack (offsetting images)
eps = 0.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
%    S_eps_02(n) = Star(lb, ub);
%    A_eps_02(n) = AbsDom(lb, ub, inf);
   RS_eps_02(n) = RStar(lb, ub, inf);
   
%    B = Box(lb,ub);
%    Z_eps_02(n) = B.toZono;
end

% save tanh_100_100_not_normalized/inputStar02.mat S_eps_02
save tanh_100_100_not_normalized/inputRStar02.mat RS_eps_02
% save tanh_100_100_not_normalized/inputAbsDom02.mat A_eps_02
% save tanh_100_100_not_normalized/inputZono02.mat Z_eps_02

eps = 0.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
%    S_eps_04(n) = Star(lb, ub);
%    A_eps_04(n) = AbsDom(lb, ub, inf);
   RS_eps_04(n) = RStar(lb, ub, inf);
   
%    B = Box(lb,ub);
%    Z_eps_04(n) = B.toZono;
end

% save tanh_100_100_not_normalized/inputStar04.mat S_eps_04
save tanh_100_100_not_normalized/inputRStar04.mat RS_eps_04
% save tanh_100_100_not_normalized/inputAbsDom04.mat A_eps_04
% save tanh_100_100_not_normalized/inputZono04.mat Z_eps_04

eps = 0.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
%    S_eps_06(n) = Star(lb, ub);
%    A_eps_06(n) = AbsDom(lb, ub, inf);
   RS_eps_06(n) = RStar(lb, ub, inf);
   
%    B = Box(lb,ub);
%    Z_eps_06(n) = B.toZono;
end

% save tanh_100_100_not_normalized/inputStar06.mat S_eps_06
save tanh_100_100_not_normalized/inputRStar06.mat RS_eps_06
% save tanh_100_100_not_normalized/inputAbsDom06.mat A_eps_06
% save tanh_100_100_not_normalized/inputZono06.mat Z_eps_06

eps = 0.8
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
%    S_eps_08(n) = Star(lb, ub);
%    A_eps_08(n) = AbsDom(lb, ub, inf);
   RS_eps_08(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_08(n) = B.toZono;
end

% save tanh_100_100_not_normalized/inputStar08.mat S_eps_08
save tanh_100_100_not_normalized/inputRStar08.mat RS_eps_08
% save tanh_100_100_not_normalized/inputAbsDom08.mat A_eps_08
% save tanh_100_100_not_normalized/inputZono08.mat Z_eps_08

eps = 1.0
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
%    S_eps_10(n) = Star(lb, ub);
%    A_eps_10(n) = AbsDom(lb, ub, inf);
   RS_eps_10(n) = RStar(lb, ub, inf);
   
%    B = Box(lb,ub);
%    Z_eps_10(n) = B.toZono;
end

% save tanh_100_100_not_normalized/inputStar10.mat S_eps_10
save tanh_100_100_not_normalized/inputRStar10.mat RS_eps_10
% save tanh_100_100_not_normalized/inputAbsDom10.mat A_eps_10
% save tanh_100_100_not_normalized/inputZono10.mat Z_eps_10

% eps = 1.2
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_12(n) = Star(lb, ub);
%    A_eps_12(n) = AbsDom(lb, ub, inf);
%    RS_eps_12(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_12(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar12.mat S_eps_12
% save tanh_100_100_not_normalized/inputRStar12.mat RS_eps_12
% save tanh_100_100_not_normalized/inputAbsDom12.mat A_eps_12
% save tanh_100_100_not_normalized/inputZono12.mat Z_eps_12
% 
% eps = 1.4
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_14(n) = Star(lb, ub);
%    A_eps_14(n) = AbsDom(lb, ub, inf);
%    RS_eps_14(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_14(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar14.mat S_eps_14
% save tanh_100_100_not_normalized/inputRStar14.mat RS_eps_14
% save tanh_100_100_not_normalized/inputAbsDom14.mat A_eps_14
% save tanh_100_100_not_normalized/inputZono14.mat Z_eps_14
% 
% eps = 1.6
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_16(n) = Star(lb, ub);
%    A_eps_16(n) = AbsDom(lb, ub, inf);
%    RS_eps_16(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_16(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar16.mat S_eps_16
% save tanh_100_100_not_normalized/inputRStar16.mat RS_eps_16
% save tanh_100_100_not_normalized/inputAbsDom16.mat A_eps_16
% save tanh_100_100_not_normalized/inputZono16.mat Z_eps_16
% 
% eps = 1.8
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_18(n) = Star(lb, ub);
%    A_eps_18(n) = AbsDom(lb, ub, inf);
%    RS_eps_18(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_18(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar18.mat S_eps_18
% save tanh_100_100_not_normalized/inputRStar18.mat RS_eps_18
% save tanh_100_100_not_normalized/inputAbsDom18.mat A_eps_18
% save tanh_100_100_not_normalized/inputZono18.mat Z_eps_18
% 
% eps = 2.0
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_20(n) = Star(lb, ub);
%    A_eps_20(n) = AbsDom(lb, ub, inf);
%    RS_eps_20(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_20(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar20.mat S_eps_20
% save tanh_100_100_not_normalized/inputRStar20.mat RS_eps_20
% save tanh_100_100_not_normalized/inputAbsDom20.mat A_eps_20
% save tanh_100_100_not_normalized/inputZono20.mat Z_eps_20
% 
% eps = 2.2
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_22(n) = Star(lb, ub);
%    A_eps_22(n) = AbsDom(lb, ub, inf);
%    RS_eps_22(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_22(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar22.mat S_eps_22
% save tanh_100_100_not_normalized/inputRStar22.mat RS_eps_22
% save tanh_100_100_not_normalized/inputAbsDom22.mat A_eps_22
% save tanh_100_100_not_normalized/inputZono22.mat Z_eps_22
% 
% eps = 2.4
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_24(n) = Star(lb, ub);
%    A_eps_24(n) = AbsDom(lb, ub, inf);
%    RS_eps_24(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_24(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar24.mat S_eps_24
% save tanh_100_100_not_normalized/inputRStar24.mat RS_eps_24
% save tanh_100_100_not_normalized/inputAbsDom24.mat A_eps_24
% save tanh_100_100_not_normalized/inputZono24.mat Z_eps_24
% 
% eps = 2.6
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_26(n) = Star(lb, ub);
%    A_eps_26(n) = AbsDom(lb, ub, inf);
%    RS_eps_26(n) = RStar(lb, ub, inf);
%    
%    B = Box(lb,ub);
%    Z_eps_26(n) = B.toZono;
% end
% 
% save tanh_100_100_not_normalized/inputStar26.mat S_eps_26
% save tanh_100_100_not_normalized/inputRStar26.mat RS_eps_26
% save tanh_100_100_not_normalized/inputAbsDom26.mat A_eps_26
% save tanh_100_100_not_normalized/inputZono26.mat Z_eps_26

% save tanh_100_100_not_normalized/inputStar02.mat S_eps_02
% save tanh_100_100_not_normalized/inputStar04.mat S_eps_04
% save tanh_100_100_not_normalized/inputStar06.mat S_eps_06
% save tanh_100_100_not_normalized/inputStar08.mat S_eps_08
% save tanh_100_100_not_normalized/inputStar10.mat S_eps_10
% save tanh_100_100_not_normalized/inputStar12.mat S_eps_12
% save tanh_100_100_not_normalized/inputStar14.mat S_eps_14
% save tanh_100_100_not_normalized/inputStar16.mat S_eps_16
% save tanh_100_100_not_normalized/inputStar18.mat S_eps_18
% save tanh_100_100_not_normalized/inputStar20.mat S_eps_20
% save tanh_100_100_not_normalized/inputStar22.mat S_eps_22
% save tanh_100_100_not_normalized/inputStar24.mat S_eps_24
% save tanh_100_100_not_normalized/inputStar26.mat S_eps_26
% 
% save tanh_100_100_not_normalized/inputAbsDom02.mat A_eps_02
% save tanh_100_100_not_normalized/inputAbsDom04.mat A_eps_04
% save tanh_100_100_not_normalized/inputAbsDom06.mat A_eps_06
% save tanh_100_100_not_normalized/inputAbsDom08.mat A_eps_08
% save tanh_100_100_not_normalized/inputAbsDom10.mat A_eps_10
% save tanh_100_100_not_normalized/inputAbsDom12.mat A_eps_12
% save tanh_100_100_not_normalized/inputAbsDom14.mat A_eps_14
% save tanh_100_100_not_normalized/inputAbsDom16.mat A_eps_16
% save tanh_100_100_not_normalized/inputAbsDom18.mat A_eps_18
% save tanh_100_100_not_normalized/inputAbsDom20.mat A_eps_20
% save tanh_100_100_not_normalized/inputAbsDom22.mat A_eps_22
% save tanh_100_100_not_normalized/inputAbsDom24.mat A_eps_24
% save tanh_100_100_not_normalized/inputAbsDom26.mat A_eps_26
% 
% save tanh_100_100_not_normalized/inputRStar02.mat RS_eps_02
% save tanh_100_100_not_normalized/inputRStar04.mat RS_eps_04
% save tanh_100_100_not_normalized/inputRStar06.mat RS_eps_06
% save tanh_100_100_not_normalized/inputRStar08.mat RS_eps_08
% save tanh_100_100_not_normalized/inputRStar10.mat RS_eps_10
% save tanh_100_100_not_normalized/inputRStar12.mat RS_eps_12
% save tanh_100_100_not_normalized/inputRStar14.mat RS_eps_14
% save tanh_100_100_not_normalized/inputRStar16.mat RS_eps_16
% save tanh_100_100_not_normalized/inputRStar18.mat RS_eps_18
% save tanh_100_100_not_normalized/inputRStar20.mat RS_eps_20
% save tanh_100_100_not_normalized/inputRStar22.mat RS_eps_22
% save tanh_100_100_not_normalized/inputRStar24.mat RS_eps_24
% save tanh_100_100_not_normalized/inputRStar26.mat RS_eps_26
% 
% save tanh_100_100_not_normalized/inputZono02.mat Z_eps_02
% save tanh_100_100_not_normalized/inputZono04.mat Z_eps_04
% save tanh_100_100_not_normalized/inputZono06.mat Z_eps_06
% save tanh_100_100_not_normalized/inputZono08.mat Z_eps_08
% save tanh_100_100_not_normalized/inputZono10.mat Z_eps_10
% save tanh_100_100_not_normalized/inputZono12.mat Z_eps_12
% save tanh_100_100_not_normalized/inputZono14.mat Z_eps_14
% save tanh_100_100_not_normalized/inputZono16.mat Z_eps_16
% save tanh_100_100_not_normalized/inputZono18.mat Z_eps_18
% save tanh_100_100_not_normalized/inputZono20.mat Z_eps_20
% save tanh_100_100_not_normalized/inputZono22.mat Z_eps_22
% save tanh_100_100_not_normalized/inputZono24.mat Z_eps_24
% save tanh_100_100_not_normalized/inputZono26.mat Z_eps_26



% eps = 0.005;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_005(n) = Star(lb, ub);
%    A_eps_005(n) = AbsDom(lb, ub, inf);
%    RS_eps_005(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.010;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_010(n) = Star(lb, ub);
%    A_eps_010(n) = AbsDom(lb, ub, inf);
%    RS_eps_010(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.015;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_015(n) = Star(lb, ub);
%    A_eps_015(n) = AbsDom(lb, ub, inf);
%    RS_eps_015(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.020;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_020(n) = Star(lb, ub);
%    A_eps_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.025;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_025(n) = Star(lb, ub);
%    A_eps_025(n) = AbsDom(lb, ub, inf);
%    RS_eps_025(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.030;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_030(n) = Star(lb, ub);
%    A_eps_030(n) = AbsDom(lb, ub, inf);
%    RS_eps_030(n) = RStar(lb, ub, inf);
% end

% eps = 0.1
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_1(n) = Star(lb, ub);
%    A_eps_1(n) = AbsDom(lb, ub, inf);
%    RS_eps_1(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.2
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_2(n) = Star(lb, ub);
%    A_eps_2(n) = AbsDom(lb, ub, inf);
%    RS_eps_2(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.3
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_3(n) = Star(lb, ub);
%    A_eps_3(n) = AbsDom(lb, ub, inf);
%    RS_eps_3(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.4
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_4(n) = Star(lb, ub);
%    A_eps_4(n) = AbsDom(lb, ub, inf);
%    RS_eps_4(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.5
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_5(n) = Star(lb, ub);
%    A_eps_5(n) = AbsDom(lb, ub, inf);
%    RS_eps_5(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.6
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_6(n) = Star(lb, ub);
%    A_eps_6(n) = AbsDom(lb, ub, inf);
%    RS_eps_6(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.7
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_7(n) = Star(lb, ub);
%    A_eps_7(n) = AbsDom(lb, ub, inf);
%    RS_eps_7(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.8
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_8(n) = Star(lb, ub);
%    A_eps_8(n) = AbsDom(lb, ub, inf);
%    RS_eps_8(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.9
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_9(n) = Star(lb, ub);
%    A_eps_9(n) = AbsDom(lb, ub, inf);
%    RS_eps_9(n) = RStar(lb, ub, inf);
% end
% 
% eps = 1.0
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_10(n) = Star(lb, ub);
%    A_eps_10(n) = AbsDom(lb, ub, inf);
%    RS_eps_10(n) = RStar(lb, ub, inf);
% end

% % save not_normalized/inputStar005.mat S_eps_005
% % save not_normalized/inputStar010.mat S_eps_010 
% % save not_normalized/inputStar015.mat S_eps_015  
% % save not_normalized/inputStar020.mat S_eps_020 
% % save not_normalized/inputStar025.mat S_eps_025 
% save not_normalized/inputStar1.mat S_eps_1
% save not_normalized/inputStar2.mat S_eps_2
% save not_normalized/inputStar3.mat S_eps_3
% save not_normalized/inputStar4.mat S_eps_4
% save not_normalized/inputStar5.mat S_eps_5
% save not_normalized/inputStar6.mat S_eps_6
% save not_normalized/inputStar7.mat S_eps_7
% save not_normalized/inputStar8.mat S_eps_8
% save not_normalized/inputStar9.mat S_eps_9
% save not_normalized/inputStar10.mat S_eps_10
% 
% % save not_normalized/inputAbsDom005.mat A_eps_005 
% % save not_normalized/inputAbsDom010.mat A_eps_010 
% % save not_normalized/inputAbsDom015.mat A_eps_015 
% % save not_normalized/inputAbsDom020.mat A_eps_020  
% % save not_normalized/inputAbsDom025.mat A_eps_025 
% % save not_normalized/inputAbsDom030.mat A_eps_030
% save not_normalized/inputAbsDom1.mat A_eps_1
% save not_normalized/inputAbsDom2.mat A_eps_2
% save not_normalized/inputAbsDom3.mat A_eps_3
% save not_normalized/inputAbsDom4.mat A_eps_4
% save not_normalized/inputAbsDom5.mat A_eps_5
% save not_normalized/inputAbsDom6.mat A_eps_6
% save not_normalized/inputAbsDom7.mat A_eps_7
% save not_normalized/inputAbsDom8.mat A_eps_8
% save not_normalized/inputAbsDom9.mat A_eps_9
% save not_normalized/inputAbsDom10.mat A_eps_10
% 
% % save not_normalized/inputRStar005.mat RS_eps_005 
% % save not_normalized/inputRStar010.mat RS_eps_010 
% % save not_normalized/inputRStar015.mat RS_eps_015 
% % save not_normalized/inputRStar020.mat RS_eps_020 
% % save not_normalized/inputRStar025.mat RS_eps_025 
% % save not_normalized/inputRStar030.mat RS_eps_030
% save not_normalized/inputRStar1.mat RS_eps_1
% save not_normalized/inputRStar2.mat RS_eps_2
% save not_normalized/inputRStar3.mat RS_eps_3
% save not_normalized/inputRStar4.mat RS_eps_4
% save not_normalized/inputRStar5.mat RS_eps_5
% save not_normalized/inputRStar6.mat RS_eps_6
% save not_normalized/inputRStar7.mat RS_eps_7
% save not_normalized/inputRStar8.mat RS_eps_8
% save not_normalized/inputRStar9.mat RS_eps_9
% save not_normalized/inputRStar10.mat RS_eps_10