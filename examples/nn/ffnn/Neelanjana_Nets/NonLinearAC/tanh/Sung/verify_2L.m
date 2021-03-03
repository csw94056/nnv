load tansig_200_50_nnv.mat;
load inputStar.mat;
load inputSet.mat
N = 50; 
numCores = 8;

% verify the network with eps = 0.5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A_eps_05(1:N) = AbsDom(S_eps_05(1:N));
RS_eps_05(1:N) = RStar(S_eps_05(1:N));

reachMethod = 'approx-star';
[r1, rb1, cE1, cands1, vt1] = net.evaluateRBN(S_eps_05(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon1 = [0.5];
verify_time1 = [sum(vt1)];
safe1 = [sum(rb1==1)];
unsafe1 = [sum(rb1 == 0)];
unknown1 = [sum(rb1 == 2)];

T1 = table(epsilon1, safe1, unsafe1, unknown1, verify_time1)
fprintf('total time star: %f ',verify_time1);
save("eps_05_star_verify_tan_2L.mat", 'T1', 'r1', 'rb1', 'cE1', 'cands1', 'vt1');


reachMethod =  'absdom';
[r2, rb2, cE2, cands2, vt2] = net.evaluateRBN(A_eps_05(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon2 = [0.5];
verify_time2 = [sum(vt2)];
safe2 = [sum(rb2==1)];
unsafe2 = [sum(rb2 == 0)];
unknown2 = [sum(rb2 == 2)];

T2 = table(epsilon2, safe2, unsafe2, unknown2, verify_time2)
fprintf('total time absdom: %f ',verify_time2);
save("eps_05_absdom_verify_tan_2L.mat", 'T2', 'r2', 'rb2', 'cE2', 'cands2', 'vt2');

reachMethod = 'rstar-absdom-two';
[r3, rb3, cE3, cands3, vt3] = net.evaluateRBN(RS_eps_05(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon3 = [0.5];
verify_time3 = [sum(vt3)];
safe3 = [sum(rb3==1)];
unsafe3 = [sum(rb3 == 0)];
unknown3 = [sum(rb3 == 2)];

T3 = table(epsilon3, safe3, unsafe3, unknown3, verify_time3)
fprintf('total time rstar: %f ',verify_time3);
save("eps_05_rstar_verify_tan_2L.mat", 'T3', 'r3', 'rb3', 'cE3', 'cands3', 'vt3');



% verify the network with eps = 0.05 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A_eps_005(1:N) = AbsDom(S_eps_005(1:N));
RS_eps_005(1:N) = RStar(S_eps_005(1:N));

reachMethod = 'approx-star';
[r4, rb4, cE4, cands4, vt4] = net.evaluateRBN(S_eps_005(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon4 = [0.05];
verify_time4 = [sum(vt4)];
safe4 = [sum(rb4==1)];
unsafe4 = [sum(rb4 == 0)];
unknown4 = [sum(rb4 == 2)];

T4 = table(epsilon4, safe4, unsafe4, unknown4, verify_time4)
fprintf('total time star: %f ',verify_time4);
save("eps_005_star_verify_tan_2L.mat", 'T4', 'r4', 'rb4', 'cE4', 'cands4', 'vt4');


reachMethod =  'absdom';
[r5, rb5, cE5, cands5, vt5] = net.evaluateRBN(A_eps_005(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon5 = [0.05];
verify_time5 = [sum(vt5)];
safe5 = [sum(rb5==1)];
unsafe5 = [sum(rb5 == 0)];
unknown5 = [sum(rb5 == 2)];

T5 = table(epsilon5, safe5, unsafe5, unknown5, verify_time5)
fprintf('total time absdom: %f ',verify_time5);
save("eps_005_absdom_verify_tan_2L.mat", 'T5', 'r5', 'rb5', 'cE5', 'cands5', 'vt5');

reachMethod = 'rstar-absdom-two';
[r6, rb6, cE6, cands6, vt6] = net.evaluateRBN(RS_eps_005(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon6 = [0.05];
verify_time6 = [sum(vt6)];
safe6 = [sum(rb6==1)];
unsafe6 = [sum(rb6 == 0)];
unknown6 = [sum(rb6 == 2)];

T6 = table(epsilon6, safe6, unsafe6, unknown6, verify_time6)
fprintf('total time rstar: %f ',verify_time6);
save("eps_005_rstar_verify_tan_2L.mat", 'T6', 'r6', 'rb6', 'cE6', 'cands6', 'vt6');

% verify the network with eps = 0.02 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A_eps_002(1:N) = AbsDom(S_eps_002(1:N));
RS_eps_002(1:N) = RStar(S_eps_002(1:N));

reachMethod = 'approx-star';
[r7, rb7, cE7, cands7, vt7] = net.evaluateRBN(S_eps_002(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon7 = [0.02];
verify_time7 = [sum(vt7)];
safe7 = [sum(rb7==1)];
unsafe7 = [sum(rb7 == 0)];
unknown7 = [sum(rb7 == 2)];

T7 = table(epsilon7, safe7, unsafe7, unknown7, verify_time7)
fprintf('total time star: %f ',verify_time7);
save("eps_002_star_verify_tan_2L.mat", 'T7', 'r7', 'rb7', 'cE7', 'cands7', 'vt7');


reachMethod =  'absdom';
[r8, rb8, cE8, cands8, vt8] = net.evaluateRBN(A_eps_002(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon8 = [0.02];
verify_time8 = [sum(vt8)];
safe8 = [sum(rb8==1)];
unsafe8 = [sum(rb8 == 0)];
unknown8 = [sum(rb8 == 2)];

T8 = table(epsilon8, safe8, unsafe8, unknown8, verify_time8)
fprintf('total time absdom: %f ',verify_time8);
save("eps_002_absdom_verify_tan_2L.mat", 'T8', 'r8', 'rb8', 'cE8', 'cands8', 'vt8');

reachMethod = 'rstar-absdom-two';
[r9, rb9, cE9, cands9, vt9] = net.evaluateRBN(RS_eps_002(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon9 = [0.02];
verify_time9 = [sum(vt9)];
safe9 = [sum(rb9==1)];
unsafe9 = [sum(rb9 == 0)];
unknown9 = [sum(rb9 == 2)];

T9 = table(epsilon9, safe9, unsafe9, unknown9, verify_time9)
fprintf('total time rstar: %f ',verify_time9);
save("eps_002_rstar_verify_tan_2L.mat", 'T9', 'r9', 'rb9', 'cE9', 'cands9', 'vt9');

% verify the network with eps = 1.2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A_eps_12(1:N) = AbsDom(S_eps_12(1:N));
RS_eps_12(1:N) = RStar(S_eps_12(1:N));

reachMethod = 'approx-star';
[r10, rb10, cE10, cands10, vt10] = net.evaluateRBN(S_eps_12(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon10 = [1.2];
verify_time10 = [sum(vt10)];
safe10 = [sum(rb10==1)];
unsafe10 = [sum(rb10 == 0)];
unknown10 = [sum(rb10 == 2)];

T10 = table(epsilon10, safe10, unsafe10, unknown10, verify_time10)
fprintf('total time star: %f ',verify_time10);
save("eps_12_star_verify_tan_2L.mat", 'T10', 'r10', 'rb10', 'cE10', 'cands10', 'vt10');


reachMethod =  'absdom';
[r11, rb11, cE11, cands11, vt11] = net.evaluateRBN(A_eps_12(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon11 = [1.2];
verify_time11 = [sum(vt11)];
safe11 = [sum(rb11==1)];
unsafe11 = [sum(rb11 == 0)];
unknown11 = [sum(rb11 == 2)];

T11 = table(epsilon11, safe11, unsafe11, unknown11, verify_time11)
fprintf('total time absdom: %f ',verify_time11);
save("eps_12_absdom_verify_tan_2L.mat", 'T11', 'r11', 'rb11', 'cE11', 'cands11', 'vt11');

reachMethod = 'rstar-absdom-two';
[r12, rb12, cE12, cands12, vt12] = net.evaluateRBN(RS_eps_12(1:N), labels(1:N), reachMethod, numCores, 0, 0, 'glpk');

epsilon12 = [1.2];
verify_time12 = [sum(vt12)];
safe12 = [sum(rb12==1)];
unsafe12 = [sum(rb12 == 0)];
unknown12 = [sum(rb12 == 2)];

T12 = table(epsilon12, safe12, unsafe12, unknown12, verify_time12)
fprintf('total time rstar: %f ',verify_time12);
save("eps_12_rstar_verify_tan_2L.mat", 'T12', 'r12', 'rb12', 'cE12', 'cands12', 'vt12');
