function [outCost,outGrad] = CostGrad(K, C , part1, part2, manifold, irho)
multiply_diag = @(A,B)  sum((A .* B') , 2); %output is diag(A * B)'
twoMatrix_trace = @(C,D) sum(multiply_diag(C,D)) ;% trace(C x D)

KCK = K * C * K;
nominator = twoMatrix_trace(KCK , C) + 2 * irho * twoMatrix_trace(C , part1) ;
denominator = 2 * irho * twoMatrix_trace(C , part2) ;
outCost = nominator / denominator ;
% outCost = nominator - denominator ;
dnom = 2* KCK +  2 * irho * part1;
ddenom =  2 * irho * part2 ;
% outGrad = dnom - ddenom ;
outGrad = dnom * denominator - ddenom * nominator;
outGrad = outGrad / denominator^2;
outGrad = manifold.egrad2rgrad(C , outGrad);
