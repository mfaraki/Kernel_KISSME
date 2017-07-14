%Install https://manopt.org/
%Install http://www.vlfeat.org/  
save_kernels;
clear; clc ;
dropbox_folder =  pwd;
symm = @(X) (X + X')/2;
dataset_name = 'iLIDS';
partition_name = 'Random';
%% load dataset partition
load([dropbox_folder '/Feature/' dataset_name '_Partition_' partition_name '.mat']);
load([dropbox_folder '/Dataset/' dataset_name '_Images.mat'], 'gID', 'camID')
Partition = Partition(1:10);
%%
np_ratio =1; % The ratio of number of negative and positive pairs. 
energy = 0.99;  %Try different values as well
for idx_partition=1:10
    load(['./matFiles/K_SVD_KKissMe_Partition' int2str(idx_partition)]);
    idx_train = Partition(idx_partition).idx_train ;
    idx_test = Partition(idx_partition).idx_test ;
    ix_train_neg_pair = Partition(idx_partition).idx_train_neg_pair;
    ix_train_pos_pair = Partition(idx_partition).idx_train_pos_pair;
    ix_test_gallery =Partition(idx_partition).ix_test_gallery;
    ix_train_gallery =Partition(idx_partition).ix_train_gallery;
    
    Nneg = min(np_ratio* length(ix_train_pos_pair), length(ix_train_neg_pair));
    ix_pair = [ix_train_pos_pair ; ix_train_neg_pair(1:Nneg,:) ]; % both positive and negative pair index
    y = [ones(size(ix_train_pos_pair,1), 1); -ones(Nneg,1)]; % annotation of positive and negative pair
    idxa = ix_pair(:,1);
    idxb = ix_pair(:,2);
    matches = y>0;
    train_sameMask = matches;
    train_differentMask = ~matches;
    Ns_tr = sum(train_sameMask);
    Nd_tr = sum(train_differentMask);
    const = 0.7071;
    Js = [const*eye(Ns_tr) -const*eye(Ns_tr); -const*eye(Ns_tr) const*eye(Ns_tr)];
    Jd = [const*eye(Nd_tr) -const*eye(Nd_tr); -const*eye(Nd_tr) const*eye(Nd_tr)];    
    K_tr_ts = K(idx_train , idx_test);    
    %      n = 150;
    %     rand_inds = randperm(length(idx_train) , n);
    %     rand_inds = idx_train(rand_inds);
    %     K_tr_rnd = K(idx_train , rand_inds);
    %     idx_train = rand_inds;
    %     K_tr_tr = K(idx_train , idx_train);
    %     Ks_tr = K_tr_rnd([idxa idxb] , :);
    %     Ks_tr = Ks_tr([train_sameMask train_sameMask],:);
    %     Kd_tr = K_tr_rnd([idxa idxb] , :);
    %     Kd_tr = Kd_tr([train_differentMask train_differentMask],:);
    %     inv_K_tr_tr = pinv(K_tr_tr);    
    
    n = length(idx_train); %Try different values as well
    K_tr_tr = K(idx_train , idx_train);
    Ks_tr = K_tr_tr([idxa idxb] , :);
    Ks_tr = Ks_tr([train_sameMask train_sameMask] , :);
    Kd_tr= K_tr_tr([idxa idxb] , :);
    Kd_tr = Kd_tr([train_differentMask train_differentMask] , :);
    inv_K_tr_tr = pinv(K_tr_tr);
    
    cnt = 1;
    diag_Ds = diag(Ds_all);
    diag_Dd = diag(Dd_all);
    energy_s = cumsum(diag_Ds) / sum(diag_Ds);
    energy_d = cumsum(diag_Dd) / sum(diag_Dd);
    for e = energy  %Try different values as well
        ind_s = find(energy_s >= e , 1);
        ind_d = find(energy_d >= e , 1);
        Vs = Vs_all(:,1:ind_s);
        Ds = Ds_all(1:ind_s,1:ind_s);
        Vd = Vd_all(:,1:ind_d);
        Dd = Dd_all(1:ind_d,1:ind_d);
        Ds_inv = 1./diag(Ds);
        Dd_inv = 1./diag(Dd);
        JVd = Jd * Vd;
        JVs = Js * Vs;
        tic
        Ad = JVd * diag((ones(ind_d,1) - 0 * Dd_inv) .* Dd_inv) * JVd';
        Ad = symm(Ad);
        As = JVs * diag((ones(ind_s,1) - 0 * Ds_inv) .* Ds_inv) * JVs';
        As = symm(As);
        for rho = 1%  2.^(power)  %Try different values as well
            [C_cf , outCost_cf] = closedform(K_tr_tr, Ks_tr, Kd_tr, As, Ad, 1/rho,inv_K_tr_tr,n);
            dist_test = cdistK_closedform(C_cf , K_tr_ts', K_tr_ts');
            rr = compute_rank_KernelKISSME(dist_test,ix_test_gallery,gID(idx_test));
            Rank = mean(rr,1);
            Rank(1);
            Results_cf(idx_partition).outCost(cnt) = outCost_cf;
            Results_cf(idx_partition).dim_s(cnt) = ind_s;
            Results_cf(idx_partition).dim_d(cnt) = ind_d ;
            Results_cf(idx_partition).rho(cnt) = rho;
            Results_cf(idx_partition).Rank(:,cnt) = Rank;
            Results_cf(idx_partition).energy_s(cnt) = energy_s(ind_s);
            Results_cf(idx_partition).energy_d(cnt) = energy_d(ind_d) ;
%             %Uncomment for the optimization method
%             [C_spd , outCost_spd] = optimize_spd(K_tr_tr, Ks_tr, Kd_tr, As, Ad, 1/rho, n);
%             dist_test = cdistK_closedform(C_spd , K_tr_ts', K_tr_ts');
%             rr = compute_rank_KernelKISSME(dist_test,ix_test_gallery,gID(idx_test));
%             Rank = mean(rr,1);            
%             Results_spd(idx_partition).outCost(cnt) = outCost_spd;
%             Results_spd(idx_partition).dim_s(cnt) = ind_s;
%             Results_spd(idx_partition).dim_d(cnt) = ind_d ;
%             Results_spd(idx_partition).rho(cnt) = rho;
%             Results_spd(idx_partition).Rank(:,cnt) = Rank;
%             Results_spd(idx_partition).energy_s(cnt) = energy_s(ind_s);
%             Results_spd(idx_partition).energy_d(cnt) = energy_d(ind_d) ;         
            cnt = cnt + 1;
        end
        toc
    end    
end
Split_Results = Results_cf;
%Split_Results = Results_spd;
temp1 = Split_Results(1).Rank;
temp2 = Split_Results(1).dim_s;
temp3 = Split_Results(1).dim_d;
temp4 = Split_Results(1).energy_s;
temp5 = Split_Results(1).energy_d;
for idx_partition=2:10
    temp1 = temp1 + Split_Results(idx_partition).Rank;
    temp2 = temp2 +  Split_Results(idx_partition).dim_s;
    temp3 = temp3 +  Split_Results(idx_partition).dim_d;
    temp4 = temp4 +  Split_Results(idx_partition).energy_s;
    temp5 = temp5 +  Split_Results(idx_partition).energy_d;
end
AVG_Results.Rank = temp1 / 10;
AVG_Results.dim_s = temp2 / 10;
AVG_Results.dim_d = temp3 / 10;
AVG_Results.energy_s = temp4 / 10;
AVG_Results.energy_d = temp5 / 10;
% save('./matFiles/Results_ClosedForm', '-v7.3', 'AVG_Results', 'Split_Results');
%save('./matFiles/Results_SPD', '-v7.3', 'AVG_Results', 'Split_Results');
