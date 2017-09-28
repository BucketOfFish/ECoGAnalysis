function [Around] = sround(A);


%
% [Around] = sround(A)

% Takes as input a matrix A with real entries and rounds all its entries
% to the nearest integer in {-1, 0, +1}. The resulting matrix is Around.
%

Around = round(A); % round to the nearest integer

% we need to handle entries that were rounded to -2,-3, etc. and/or 2,3,etc.
[x,y] = find(abs(Around) > 1); % find all entries larger than +1 or -1

for i=1:size(x,1),
    
    if Around(x(i),y(i)) > 1   % if larger than 1, decrease it to 1
        Around(x(i),y(i)) = 1;
    elseif Around(x(i),y(i)) < -1
        Around(x(i),y(i)) = -1; % if smaller than -1, increase it to -1
    end
    
end % end for