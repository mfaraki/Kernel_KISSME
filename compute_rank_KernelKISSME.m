function Ranks = compute_rank_KernelKISSME(dist_test,ix_partition,IDs)
for k = 1:size(ix_partition,1)
    ix_ref = ix_partition(k,:) == 1;
    ix_prob = ~ix_ref;
    
    ref_ID = IDs(ix_ref);
    prob_ID = IDs(ix_prob);
    
    N_ref = length(ref_ID);
    %N_prob = length(prob_ID);
    
    dist = dist_test(ix_prob, ix_ref);
%     for p = 1:N_prob
%         [~, ix] = sort(dist(p, :));
%         r(p) =  find(ref_ID(ix) == prob_ID(p));     
%     end
    
    [~, ixs] = sort(dist');    
    prob_ID_rep = repmat(prob_ID, N_ref, 1);  
    [r , ~] = find( ref_ID(ixs')' == prob_ID_rep);
    R(k, :) = r;    
end

[a, ~] = hist(R',1 : N_ref);
Ranks = cumsum(a) /length(prob_ID);
Ranks = Ranks';
