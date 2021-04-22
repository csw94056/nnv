close all;
clear;
clc;
load RS_eps_16_absdom_reach.mat;
load S_eps_16_abs_dom_reach.mat

R = RS_eps_16_absdom_reach;
S = S_eps_16_abs_dom_reach;
%check image 3 

RC = R(3).C;
SC = S(3).C;

save('RC.csv', 'RC', '-ascii', '-double', '-tabs');
save('SC.csv', 'SC', '-ascii', '-double', '-tabs');