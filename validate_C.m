function Chat = validate_C(C)
[eig_vectors,eig_values] = eig(C);
diag_eig = diag(eig_values);
[diag_eig, ind] = sort(diag_eig, 'descend');
diag_eig(diag_eig <= 1e-10) = 1e-6;
eig_values = diag(diag_eig);
eig_vectors = eig_vectors(:,ind);
Chat = eig_vectors * eig_values * eig_vectors';