function [C , outCost] = closedform(K_tr_tr, Ks_tr, Kd_tr, As, Ad, irho,inv_K_tr_tr, n)
symm = @(X) (X + X')/2;
part1 = Kd_tr' * Ad * Kd_tr;
part1 = symm(part1);
part2 = Ks_tr' * As * Ks_tr ;
part2 = symm(part2);
C = irho * (part1 - part2);
C = symm(C);
C = inv_K_tr_tr * C * inv_K_tr_tr;
C = symm(C);
C = validate_C(C);
C = symm(C);
outCost = nan;
%manifold = sympositivedefinitefactory(n);
%[outCost,~]  = CostGrad_relational_spd(K_tr_tr, C/norm(C,'fro'), Ks_tr, Kd_tr, As, Ad, manifold,irho);

