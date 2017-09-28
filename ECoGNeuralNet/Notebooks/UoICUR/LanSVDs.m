  function [s, v, u, k] = LanSVDs(X, num, y, z, stride, tol, msteps, flag)
%
%  Approximate extreme singular values/vectors by Lanczos method.
%  Usage: [s, V, U, niter] = LanSVDs(X, num, y, z, stride, tol, msteps, flag)
%
%  Input:
%  X: the data matrix of size m-by-n (m>=n).
%  y,z are for low rank correction, useful when X is sparse.
%  (i.e., singular values of X+y*z are approximated rather than X.)
%  By default, y,z are zero vectors; use [],[] to bypass them.
%  y,z are helpful for PCA computation when the data matrix is sparse.
%  To be precise, let X be the data matrix (columns are data entries).
%  Then set y=sum(X,2)/n and z=ones(1,m) for the PCA computation.
%
%  num: number of extreme singular values/vectors approximated (default 1).
%  stride: how often the convergence is tested (default min(num,20)).
%  tol: tolerance (nonnegative, can be zero).
%  msteps: the maximum number of Lanczos steps (default n).
%  flag is the 3rd input argument of Matlab built-in routine svds (try
%  'help svds'). By default, flag='L' for largest singular values/vectors.
%  For smallest ones, set flag=0.
%
%  Output:
%  s: extreme singular values (in a column vector).
%  V: the corresponding right singular vectors (in a matrix).
%  U: the corresponding left singular vectors (in a matrix).
%  niter: number of Lanczos iterations.
%
%  In one word, U*diag(s)*V approximates X-y*z.


verbose = 1;
[m,n] = size(X);
% num = min(num,min(m,n));
T = sparse(1,1);
memory_size = 5*num;
V = zeros(n,memory_size);
expand_factor = 1.2;
if ~exist('y','var') | size(y,1)~=m
    y = zeros(m,1);
end
if ~exist('z','var') | size(z,2)~=n
    z = zeros(1,n);
end
if ~exist('num','var') | length(num)==0
    num = 1;
end
if ~exist('stride','var') | length(stride)==0
    if num <= 2
        stride = 1;
    else
        stride = min(num, 50);
    end
end
if ~exist('tol','var') | length(tol)==0
    tol = 8*eps*(norm(X,inf)+norm(y,inf)+norm(z,inf));
end
if ~exist('msteps','var') | length(msteps)==0
    msteps = min(m,n);
end
if ~exist('flag','var') | flag~=0
    flag = 'L';  % For largest singular vectors.
end

v = randn(n,1);
v = sign(v);
v = v/norm(v,2);
beta = 0.0;
V(:,1) = v;
vold = v;
for k = 1:msteps
    % w = (X+y*z)'*((X+y*z)*v);
    % For efficiency when X is sparse, (X+y*z)*v is computed by
    % X*v+y*(z*v). Also avoid X' (transpose of X).
    w = X*v+y*(z*v);
    w = (w'*X+(w'*y)*z)';
    w = w - beta*vold;  % For numerical stability.
    alpha = w'*v;
    T(k,k) = alpha;
    if k == msteps  % Last step.
        break;
    end
    if k==num & stride==1
        ss = eig(full(T));
    end
    if k>num & mod(k-num,stride)==0
        if stride==1
            ss_old = ss;
        else
            ss_old = eig(full(T(1:k-1,1:k-1)));
        end
        ss = eig(full(T));
        if flag == 'L'
            if ss(k-num+1)-ss_old(k-num) < tol
                break;
            end
        elseif ss_old(num)-ss(num) < tol
            break;
        end
        % Normally loop breaks in the above if-then-else statement.
    end
    w = w - alpha*v;
    w = w - V(:,1:k)*(V(:,1:k)'*w);  % Full reorthogonalization.
    beta = w'*w;
    beta = sqrt(beta);
    if beta < tol
        break;
    end
    vold = v;
    v = w/beta;
    if k == memory_size
        % Expand V..
        memory_size = ceil(k*expand_factor);
        V(:,k+1:memory_size) = 0;
    end
    V(:,k+1) = v;
    T(k,k+1) = beta;
    T(k+1,k) = beta;
    if verbose & mod(k,100)==0
        fprintf('.');
    end
end
if verbose
    fprintf(' SVD niter = %d\n', k);
end

[V2,D] = eig(full(T));
[s,idx] = sort(diag(D));
V2 = V2(:,idx);
if flag == 'L'  % Largest singular values.
    s = s(k:-1:k-num+1);
else  % Smallest singular values.
    s = s(1:num);
end
s = s.^0.5;
if nargout >= 2
    if flag == 'L'
        v = V(:,1:k)*V2(:,k:-1:k-num+1);
    else
        v = V(:,1:k)*V2(:,1:num);
    end
end
if nargout >= 3
    u = full(X*v + y*(z*v));
    u = full(u*spdiags(1./s,0,num,num));
end
