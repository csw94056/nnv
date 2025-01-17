load tansig_200_50_nnv.mat;
load inputStar.mat;
load inputSet.mat
N = 25; 
numCores = 6;
reachMethod = 'approx-star';
% verify the network with eps = 5

figure('Name', 'S')                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 36 samples
    subplot(5,10,i)                              % plot them in 6 x 6 grid
    digit = reshape(S_eps_05(i).state_lb, [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(labels(i))                   % show the label
end

% [r1, rb1, cE1, cands1, vt1] = net.evaluateRBN(S_eps_005(1:N), labels(1:N), reachMethod, numCores);
% 
% 
% % verify the network with eps = 12
% %[r2, rb2, cE2, cands2, vt2] = net.evaluateRBN(S_eps_12(1:N), labels(1:N), reachMethod, numCores);
% 
% epsilon = [0.02];
% verify_time = [sum(vt1)];
% safe = [sum(rb1==1)];
% unsafe = [sum(rb1 == 0)];
% unknown = [sum(rb1 == 2)];
% 
% % buid table 
% %epsilon = [0.02; 0.05];
% %verify_time = [sum(vt1); sum(vt2)];
% %safe = [sum(rb1==1); sum(rb2 == 1)];
% %unsafe = [sum(rb1 == 0); sum(rb2 == 0)];
% %unknown = [sum(rb1 == 2); sum(rb2 == 2)];
% 
% T = table(epsilon, safe, unsafe, unknown, verify_time)
% fprintf('total time: %f ',verify_time);
% save("/results/verify_tan_2L.mat", 'T', 'r1', 'rb1', 'cE1', 'cands1', 'vt1');
% %,  'r2', 'rb2', 'cE2' 'cands2', 'vt2');
% 
