function dist = cdistK_closedform(C_hat , K_P_tr, K_Q_tr)
multiply_diag = @(A,B)  sum((A .* B') , 2); % A and B be two matrices to be diagonally multiplied (output is diag(A * B)')
% dist = diag(K_P_tr * C_hat * K_P_tr' -  K_P_tr * C_hat * K_Q_tr'   -  K_Q_tr * C_hat * K_P_tr' +  K_Q_tr * C_hat * K_Q_tr');
% dist = diag(K_P_tr * C_hat * K_P_tr' -  2 * K_P_tr * C_hat * K_Q_tr'  +  K_Q_tr * C_hat * K_Q_tr');

% for i=1:size(K_P_tr,1)
%     a1 = K_P_tr(i,:) * C_hat * K_P_tr(i,:)';
%     a2 = K_P_tr(i,:) * C_hat * K_Q_tr(i,:)';
%     a3 = K_Q_tr(i,:) * C_hat * K_P_tr(i,:)';
%     a4 = K_Q_tr(i,:) * C_hat * K_Q_tr(i,:)';
%     dist2(i) = a1 - a2 - a3 + a4;
% end
N_P = size(K_P_tr,1);
N_Q = size(K_Q_tr,1);

K_P_C = K_P_tr * C_hat;
K_Q_C = K_Q_tr * C_hat;

u1 =  multiply_diag(K_P_C , K_P_tr') ;
u2 =  K_P_C *  K_Q_tr' ;
u4 = multiply_diag(K_Q_C ,  K_Q_tr') ;
dist =  repmat(u1 , 1, N_Q) + repmat(u4', N_P, 1) - 2 *u2;
dist(dist < 0 ) = 0;

% for i=1:size(K_P_tr,1)
%     a1 = K_P_tr(i,:) * C_hat * K_P_tr(i,:)';
%     for j=1:size(K_Q_tr,1)
%         if i <= j
%             a2 = K_P_tr(i,:) * C_hat * K_Q_tr(j,:)';
%             %a3 = K_Q_tr(j,:) * C_hat * K_P_tr(i,:)';
%             a4 = K_Q_tr(j,:) * C_hat * K_Q_tr(j,:)';
%             dist2(i,j) = a1 - 2 *a2 + a4;
%             dist2(j,i) = dist2(i,j);
%         end
%     end
% end