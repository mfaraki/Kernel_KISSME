%Install http://www.vlfeat.org/  first
clear; clc;
dropbox_folder =  pwd;
dataset_name = 'iLIDS'; %
num_patch = 6; %6, 14, 75, 341
partition_name = 'Random';
fname = [dataset_name '_HistMoment' num2str(num_patch) 'Patch_woPreFiltering.mat'];
load([dropbox_folder '/Feature/' fname]);
featurename = fieldnames(DataSet.idxfeature);
idx_feat =[];
for i = 1: 9
    if strcmp(featurename{i}, 'HSVv')
        continue;
    end
    temp = getfield(DataSet.idxfeature, featurename{i});
    idx_feat = [idx_feat; temp(:)];
end
% LBP histogram
for i = 10: 21%length(featurename)
    if 	strcmp(featurename{i},'n8u2r1') || strcmp(featurename{i},'n16u2r2')
        temp = getfield(DataSet.idxfeature, featurename{i});
        idx_feat = [idx_feat; temp(:)];
    end
end
X = double(DataSet.data(:, idx_feat)');
clear DataSet;
%% load dataset partition
load([dropbox_folder '/Feature/' dataset_name '_Partition_' partition_name '.mat']);
load([dropbox_folder '/Dataset/' dataset_name '_Images.mat'], 'gID', 'camID')
Partition = Partition(1:10);%%
% The number of test times with the same train/test partition.
%addpath(genpath('F:\Toolboxes\toolbox'));
% K = ComputeKernel(X' , kernel_name);

K = vl_alldist2(X,X,'KCHI2');
np_ratio =1;
for idx_partition=1:10%:length(Partition) % partition loop    
    idx_train = Partition(idx_partition).idx_train ;
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
    K_tr_tr = K(idx_train , idx_train);
    K_tr_a_b = K_tr_tr([idxa idxb] , [idxa idxb]);
    
    Ks_s = K_tr_a_b([train_sameMask train_sameMask],[train_sameMask train_sameMask]);
    Kd_d = K_tr_a_b([train_differentMask train_differentMask],[train_differentMask train_differentMask]);
    [Vs_all , Ds_all , ~] = svd(Js' * Ks_s * Js);
    [Vd_all , Dd_all , ~] = svd(Jd' * Kd_d * Jd);
    save(['./matFiles/K_SVD_KKissMe_Partition' int2str(idx_partition)], '-v7.3', 'K' , 'Vs_all' , 'Ds_all' ,'Vd_all' , 'Dd_all' );
end