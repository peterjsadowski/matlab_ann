% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

function mnistdisp(datamat, datamat2)
% display a group of MNIST images 

if nargin == 1
    imgs1 = vec2imgs(datamat);
    imagesc(imgs1,[0 1]); colormap gray; axis equal; axis off;
else
    imgs1 = vec2imgs(datamat);
    imgs2 = vec2imgs(datamat2);
    imagesc([imgs1; imgs2],[0 1]); colormap gray; axis equal; axis off;
end
%drawnow;
end

function total = vec2imgs(vecs)
col=28;
row=28;

[N, nfeat] = size(vecs);
assert(nfeat == col*row);

img = cell(1, N);
total = [];
for i = 1:N
    img{i} = reshape(vecs(i, :), row, col)';
    if i == 1
        total = img{1};
    else
        total = [total, img{i}]; %#ok<AGROW>
    end
end
end
