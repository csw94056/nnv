load outputSet.mat;
load ACASXU_run2a_3_7_batch_2000.mat;


normalized_mat = range_for_scaling(6) * eye(5);
normalized_vec = means_for_scaling(6) * ones(5,1);



% normalize output set

fprintf('\nNormalize output set');

t = tic;
n = length(R0);
R0_norm = [];
parfor i=1:n
    fprintf('\nNormalize %d^th exact polyhedron reach set', i);
    R0_norm = [R0_norm  R0(i).affineMap(normalized_mat) + normalized_vec]; % exact normalized reach set
end
normalized_time0 = toc(t);

t = tic;
n = length(R1);
R1_norm = [];
parfor i=1:n
    fprintf('\nNormalize %d^th exact star reach set', i);
    R1_norm = [R1_norm  R1(i).affineMap(normalized_mat, normalized_vec)]; % exact normalized reach set
end
normalized_time1 = toc(t);




% output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
% safety property: COC is not the minimal score
% unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1
% <= x5

unsafe_mat = [1 -1 0 0 0; 1 0 -1 0 0; 1 0 0 -1 0; 1 0 0 0 -1];
unsafe_vec = [0; 0; 0; 0];


t = tic;
fprintf('\nVerifying exact polyhedron reach set...');
safe = 0;
n = length(R0);
parfor i=1:n
    R = Conversion.toStar(R0_norm(i));
    S = R.intersectHalfSpace(unsafe_mat, unsafe_vec);
    if isempty(S)
        fprintf('\nThe %d^th polyhedron output set does not reaches the unsafe region, P3 holds', i);
    else
        fprintf('\nThe %d^th polyhedron output set reaches the unsafe region, P3 is violated', i);
        safe = safe + 1;
    end
end

if safe >= 1
    fprintf('\nP3 is violated on N3_7');
else
    fprintf('\nP3 holds on N3_7');
end
exact_polyhedron_safety_checking_time = toc(t) + normalized_time0;
fprintf('\nFinish verifying exact polyhedron reach set in %.4f seconds.', exact_polyhedron_safety_checking_time);


t = tic;
fprintf('\nVerifying exact star reach set...');
safe = 0;
n = length(R1);
parfor i=1:n
    S = R1_norm(i).intersectHalfSpace(unsafe_mat, unsafe_vec);
    if isempty(S)
        fprintf('\nThe %d^th star output set does not reaches the unsafe region, P3 holds', i);
    else
        fprintf('\nThe %d^th star output set reaches the unsafe region, P3 is violated', i);
        safe = safe + 1;
    end
end

if safe >= 1
    fprintf('\nP3 is violated on N3_7');
else
    fprintf('\nP3 holds on N3_7');
end

exact_star_safety_checking_time = toc(t) + normalized_time1;
fprintf('\nFinish verifying exact star reach set in %.4f seconds.', exact_star_safety_checking_time);

t = tic;
fprintf('\nVerifying over-approximate star reach set...');
R2_norm = R2.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using star
S = R2_norm.intersectHalfSpace(unsafe_mat, unsafe_vec);
if isempty(S)
    fprintf('\nThe over-approximate star reach set does not reaches the unsafe region, P3 holds on N3_7');
else
    fprintf('\nThe over-approximate star reach set reaches the unsafe region, safety is unknown, P3 may be violated on N3_7');
end

approx_star_safety_checking_time = toc(t);

fprintf('\nFinish verifying over-approximate star reach set in %.4f seconds.', approx_star_safety_checking_time);


t = tic;
fprintf('\nVerifying over-approximate zonotope reach set...');
R3_norm = R3.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using zonotope
R = R3_norm.toStar;
S = R.intersectHalfSpace(unsafe_mat, unsafe_vec);
if isempty(S)
    fprintf('\nThe over-approximate zonotope reach set does not reaches the unsafe region, P3 holds on N3_7');
else
    fprintf('\nThe over-approximate zonotope reach set reaches the unsafe region, safety is unknown, P3 may be violated on N3_7');
end

approx_zono_safety_checking_time = toc(t);

fprintf('\nFinish verifying over-approximate zonotope reach set in %.4f seconds.', approx_zono_safety_checking_time);


t = tic;
fprintf('\nVerifying over-approximate star abstract-domain reach set...');
R4_norm = R4.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using abstract domain
S = R4_norm.intersectHalfSpace(unsafe_mat, unsafe_vec);
if isempty(S)
    fprintf('\nThe over-approximate star abstract-domain reach set does not reaches the unsafe region, P3 holds on N3_7');
else
    fprintf('\nThe over-approximate star abstract-domain reach set reaches the unsafe region, safety is unknown, P3 may be violated on N3_7');
end

approx_star_abs_dom_safety_checking_time = toc(t);

fprintf('\nFinish verifying over-approximate star abstract-domain reach set in %.4f seconds.', approx_star_abs_dom_safety_checking_time);


t = tic;
fprintf('\nVerifying over-approximate AbsDom adstract-domain reach set...');
R5_norm = R5.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using absdom (abstract-domain)
R = R5_norm.toStar;
S = R.intersectHalfSpace(unsafe_mat, unsafe_vec);
if isempty(S)
    fprintf('\nThe over-approximate AbsDom abstract-domain reach set does not reaches the unsafe region, P3 holds on N3_7');
else
    fprintf('\nThe over-approximate AbsDom abstract-domain reach set reaches the unsafe region, safety is unknown, P3 may be violated on N3_7');
end

approx_absdom_safety_checking_time = toc(t);

fprintf('\nFinish verifying over-approximate AbsDom abstract-domain reach set in %.4f seconds.', approx_absdom_safety_checking_time);


t = tic;
fprintf('\nVerifying over-approximate rstar abstract-domain with two predicate constraint reach set...');
R6_norm = R6.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using rstar (relaxed star) abstract-domain with two predicate constraint
S = R6_norm.intersectHalfSpace(unsafe_mat, unsafe_vec);
if isempty(S)
    fprintf('\nThe over-approximate rstar abstract-domain with two predicate constraint reach set does not reaches the unsafe region, P3 holds on N3_7');
else
    fprintf('\nThe over-approximate rstar abstract-domain with two predicate constraint reach set reaches the unsafe region, safety is unknown, P3 may be violated on N3_7');
end

approx_rstar_absdom_two_safety_checking_time = toc(t);

fprintf('\nFinish verifying over-approximate AbsDom abstract-domain reach set in %.4f seconds.', approx_rstar_absdom_two_safety_checking_time);


t = tic;
fprintf('\nVerifying over-approximate rstar abstract-domain with three predicate constraint reach set...');
R7_norm = R7.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using rstar (relaxed star) abstract-domain with three predicate constraint
S = R7_norm.intersectHalfSpace(unsafe_mat, unsafe_vec);
if isempty(S)
    fprintf('\nThe over-approximate rstar abstract-domain with three predicate constraint reach set does not reaches the unsafe region, P3 holds on N3_7');
else
    fprintf('\nThe over-approximate rstar abstract-domain with three predicate constraint reach set reaches the unsafe region, safety is unknown, P3 may be violated on N3_7');
end
 
approx_rstar_absdom_three_safety_checking_time = toc(t);

fprintf('\nFinish verifying over-approximate AbsDom abstract-domain reach set in %.4f seconds.', approx_rstar_absdom_three_safety_checking_time);

save safety_checking_time.mat exact_polyhedron_safety_checking_time exact_star_safety_checking_time approx_star_safety_checking_time approx_zono_safety_checking_time approx_star_abs_dom_safety_checking_time approx_absdom_safety_checking_time approx_rstar_absdom_two_safety_checking_time approx_rstar_absdom_three_safety_checking_time