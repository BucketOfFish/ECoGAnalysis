clc
clear all
close all;

%% Load data
load 'DATA.mat'

X = F;   % input matrix


[m, n] = size(X);  % input size
s=4;   %% number of bootstraps

%% methods
i=1;

k_sv = 50; % input rank
    %% ---UoICUR
    %for s=1:8
    [mdlstrct] = CUR_UoI(X,k_sv,s);
    indx=mdlstrct.opt.indx;
    C1=X(:,indx);     %% Columns from UoICUR
    [~,c1]=size(C1);
    
    %%--error for UoICUR
    Xapp1 = sround(C1*pinv(C1)*X);   % project X on C and
    %round the result (this is our approximation to X)
    err1(i,1) = nnz(Xapp1 - X)/numel(X)
 
    
    %% ---basicCUR
    [s1, V1, U1] = LanSVDs(X,k_sv);
    [C,indx2] = ColumnSelect(X, c1, c1+20, V1);
    C2=C(:,1:c1);  %% Columns from basicCUR
    
     %%--error for basicCUR
    Xapp2 = sround(C2*pinv(C2)*X);
    err2(i,1) = nnz(Xapp2 - X)/numel(X)
     % e2(i,1)=norm(X-U2*(U2'*X),'fro')
    
%      %% ---greedyCUR
%     [SNP_ind] = greedyrows(X', c1); % pick columns of A greedily
%      C3 = X(:,SNP_ind);  %% Columns greedy CUR
%     Xapp3 = sround(C3*pinv(C3)*X);
%     err3(i,1) = nnz(Xapp3- X)/numel(X);

%% Plots
% plot(sz,err1,'r*-')
% hold on;
% plot(sz,err2,'bo-')
% plot(sz,err3,'gs-')
% legend('UoI-CUR','basic-CUR','Greedy-CUR')
% %xlabel('Number of bootstraps B1-->')
% xlabel('Number of colmuns c-->')
% %ylabel('Frobenius norm error ||A-PcA||_F')
% ylabel('Error')
% title(sprintf('%d,',sz))
