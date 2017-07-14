function [C , outCost] = optimize_spd(K_tr_tr, K_Zs_trn, K_Zd_trn, As, Ad, irho, n)
 symm = @(X) (X + X')/2;
manifold = sympositivedefinitefactory(n);
problem.M = manifold;
part1 = K_Zs_trn' * As * K_Zs_trn;
part2 = K_Zd_trn' * Ad * K_Zd_trn;
problem.costgrad = @(C) CostGrad(K_tr_tr, C, part1, part2, manifold,irho);
% checkgradient(problem);
options.maxiter = 30;  %Try different values as well
% options.verbosity = 0;
C_init = eye(n);
[C,outCost]  = conjugategradient(problem, C_init, options);
C = validate_C(C);
C = symm(C);
% [outCost,~]  = CostGrad_relational_spd(K_tr_tr, C/norm(C,'fro'), Ks_tr, Kd_tr, As, Ad, manifold,irho);