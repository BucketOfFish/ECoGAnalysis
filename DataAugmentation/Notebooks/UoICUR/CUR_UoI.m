function [mdlstrct] = CUR_UoI(A,maxk,s)
%[mdlstrct] = CUR_UoI(A,maxk,s)
%Implements the UoI_CUR algorithm using bootstrap subsampling and leverage scores sampling of colmuns
% Only does column selection. For row selection input A'
%A: Input  matrix: n instances X d variables
%maxk: maximum rank k for the decomposition
%s = number of bootstrap samples in selection

verb=1; %verbose display of iterations
[n,d]=size(A);
%% parameters for the procedure
mdlstrct.parms.nbootS = s; %number of bootstrap samples 
mdlstrct.parms.rndfrctL = .9; %fraction of data used
mdlstrct.parms.nMP = 1; %number of different rank
mdlstrct.parms.k0 = maxk; %range of rank

tic
%% main loop over different rank k 
for i = 1:mdlstrct.parms.nMP %
    k=mdlstrct.parms.k0(i);
    %% For different bootstraps, generate samples and compute CUR
    for c = 1:mdlstrct.parms.nbootS
        if verb
            if mod(c,10)==0; disp(c); end
        end
        %the bootstrap sample
        rndsd = randperm(n); rndsd = rndsd(1:round(mdlstrct.parms.rndfrctL*n)); 
          [s1, V1, U1] = LanSVDs(A(rndsd,:),k);  % Compute the rank k SVD (LanSVDs a faster version of svds)
          [C,indx1] = ColumnSelect(A(rndsd,:), k, k+20, V1); % sample columns using the leverage scores
          idx{c} = indx1;  %save the indices
        randind(:,c)=rndsd;
    end
   
    
 %% Do the intersection operation    
 for c = 1:mdlstrct.parms.nbootS
            indx1 = idx{c};
            if c==1, intw = indx1; end
            intw = intersect(intw,indx1); %sintersection of indices across bootstrap samples
 end
    mdlstrct.rank(i).indx=intw;
end

%% Union of indices
%%% union of indices over different ranks k
uind=[];
for i=1:mdlstrct.parms.nMP %for each rank
      indx1= mdlstrct.rank(i).indx;
      uind=[uind,indx1];    % union of indices over different ranks
      uind=unique(uind);     % get the unique indices
end
mdlstrct.opt.indx = uind;%

toc
