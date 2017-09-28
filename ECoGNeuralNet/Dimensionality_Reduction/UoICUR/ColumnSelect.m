function  [C,indexA] = ColumnSelect(A, k, c, v) 
%
% Input
%   - A: m x n matrix.
%   - k: rank parameter k.
%   - c: number of columns that we want to select from A.
%   - v: n x k matrix of the top-k right singular vectors of A.

% Output
%   - C: m x c' matrix with c' columns from A, E(c') <= c.

[m n] = size(A)  ; % the size of the input matrix A.

%------- Compute the normalized leverage scores of eqn. 3 of [1]. ---------
pi = zeros(1, n) ; 
for j=1:n
    pi(j) =  (norm(v(j,:))^2) / k  ;
end
%--------------------------------------------------------------------------

%---------------- randomized column selection -----------------------------

indexA = []; % indexA is initially empty. 

for j=1:n    % for every column of A
    
    % the j-th column of A is selected with probability prob_j.
    prob_j = min([1 c*pi(j)]);  % find the minimum of 1 and  c*pi(j)
    prob_j = prob_j(1);         % resolve the case where 1 = c*pi(j)
        
    if prob_j==1             % if prob_j=1 select the j-th column of A
        indexA = [indexA j];
    elseif  prob_j > rand    % if prob_j<1, generate a random number rand in [0,1] and then 
        indexA = [indexA j]; % if prob_j > rand, select the j-th column of A 
    end
    
end

% At the end of this process indexA contains the indices of the selected 
% columns of A, i.e. C = A(:, indexA);

C = A(:, indexA);
return
%