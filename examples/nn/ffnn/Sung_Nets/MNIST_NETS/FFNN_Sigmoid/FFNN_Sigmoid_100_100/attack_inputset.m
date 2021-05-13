load sigmoid_100_100_images.mat;
load MNIST_sigmoid_100_100_DenseNet.mat;
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
   S_eps_02(n) = Star(lb, ub);
   A_eps_02(n) = AbsDom(lb, ub, inf);
   RS_eps_02(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_02(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar02.mat S_eps_02
save sigmoid_100_100_not_normalized/inputRStar02.mat RS_eps_02
save sigmoid_100_100_not_normalized/inputAbsDom02.mat A_eps_02
save sigmoid_100_100_not_normalized/inputZono02.mat Z_eps_02

eps = 0.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_04(n) = Star(lb, ub);
   A_eps_04(n) = AbsDom(lb, ub, inf);
   RS_eps_04(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_04(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar04.mat S_eps_04
save sigmoid_100_100_not_normalized/inputRStar04.mat RS_eps_04
save sigmoid_100_100_not_normalized/inputAbsDom04.mat A_eps_04
save sigmoid_100_100_not_normalized/inputZono04.mat Z_eps_04

eps = 0.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_06(n) = Star(lb, ub);
   A_eps_06(n) = AbsDom(lb, ub, inf);
   RS_eps_06(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_06(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar06.mat S_eps_06
save sigmoid_100_100_not_normalized/inputRStar06.mat RS_eps_06
save sigmoid_100_100_not_normalized/inputAbsDom06.mat A_eps_06
save sigmoid_100_100_not_normalized/inputZono06.mat Z_eps_06

eps = 0.8
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_08(n) = Star(lb, ub);
   A_eps_08(n) = AbsDom(lb, ub, inf);
   RS_eps_08(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_08(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar08.mat S_eps_08
save sigmoid_100_100_not_normalized/inputRStar08.mat RS_eps_08
save sigmoid_100_100_not_normalized/inputAbsDom08.mat A_eps_08
save sigmoid_100_100_not_normalized/inputZono08.mat Z_eps_08

eps = 1.0
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_10(n) = Star(lb, ub);
   A_eps_10(n) = AbsDom(lb, ub, inf);
   RS_eps_10(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_10(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar10.mat S_eps_10
save sigmoid_100_100_not_normalized/inputRStar10.mat RS_eps_10
save sigmoid_100_100_not_normalized/inputAbsDom10.mat A_eps_10
save sigmoid_100_100_not_normalized/inputZono10.mat Z_eps_10

eps = 1.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_12(n) = Star(lb, ub);
   A_eps_12(n) = AbsDom(lb, ub, inf);
   RS_eps_12(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_12(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar12.mat S_eps_12
save sigmoid_100_100_not_normalized/inputRStar12.mat RS_eps_12
save sigmoid_100_100_not_normalized/inputAbsDom12.mat A_eps_12
save sigmoid_100_100_not_normalized/inputZono12.mat Z_eps_12

eps = 1.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_14(n) = Star(lb, ub);
   A_eps_14(n) = AbsDom(lb, ub, inf);
   RS_eps_14(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_14(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar14.mat S_eps_14
save sigmoid_100_100_not_normalized/inputRStar14.mat RS_eps_14
save sigmoid_100_100_not_normalized/inputAbsDom14.mat A_eps_14
save sigmoid_100_100_not_normalized/inputZono14.mat Z_eps_14

eps = 1.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_16(n) = Star(lb, ub);
   A_eps_16(n) = AbsDom(lb, ub, inf);
   RS_eps_16(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_16(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar16.mat S_eps_16
save sigmoid_100_100_not_normalized/inputRStar16.mat RS_eps_16
save sigmoid_100_100_not_normalized/inputAbsDom16.mat A_eps_16
save sigmoid_100_100_not_normalized/inputZono16.mat Z_eps_16

eps = 1.8
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_18(n) = Star(lb, ub);
   A_eps_18(n) = AbsDom(lb, ub, inf);
   RS_eps_18(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_18(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar18.mat S_eps_18
save sigmoid_100_100_not_normalized/inputRStar18.mat RS_eps_18
save sigmoid_100_100_not_normalized/inputAbsDom18.mat A_eps_18
save sigmoid_100_100_not_normalized/inputZono18.mat Z_eps_18

eps = 2.0
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_20(n) = Star(lb, ub);
   A_eps_20(n) = AbsDom(lb, ub, inf);
   RS_eps_20(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_20(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar20.mat S_eps_20
save sigmoid_100_100_not_normalized/inputRStar20.mat RS_eps_20
save sigmoid_100_100_not_normalized/inputAbsDom20.mat A_eps_20
save sigmoid_100_100_not_normalized/inputZono20.mat Z_eps_20

eps = 2.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_22(n) = Star(lb, ub);
   A_eps_22(n) = AbsDom(lb, ub, inf);
   RS_eps_22(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_22(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar22.mat S_eps_22
save sigmoid_100_100_not_normalized/inputRStar22.mat RS_eps_22
save sigmoid_100_100_not_normalized/inputAbsDom22.mat A_eps_22
save sigmoid_100_100_not_normalized/inputZono22.mat Z_eps_22

eps = 2.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_24(n) = Star(lb, ub);
   A_eps_24(n) = AbsDom(lb, ub, inf);
   RS_eps_24(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_24(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar24.mat S_eps_24
save sigmoid_100_100_not_normalized/inputRStar24.mat RS_eps_24
save sigmoid_100_100_not_normalized/inputAbsDom24.mat A_eps_24
save sigmoid_100_100_not_normalized/inputZono24.mat Z_eps_24

eps = 2.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_26(n) = Star(lb, ub);
   A_eps_26(n) = AbsDom(lb, ub, inf);
   RS_eps_26(n) = RStar(lb, ub, inf);

   B = Box(lb,ub);
   Z_eps_26(n) = B.toZono;
end

save sigmoid_100_100_not_normalized/inputStar26.mat S_eps_26
save sigmoid_100_100_not_normalized/inputRStar26.mat RS_eps_26
save sigmoid_100_100_not_normalized/inputAbsDom26.mat A_eps_26
save sigmoid_100_100_not_normalized/inputZono26.mat Z_eps_26
