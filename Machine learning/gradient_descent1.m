function [X0, iter] = GD()

f = @(x,y) x.^2 + y.^2 - 2*y + 3*y;
% symbolic function to be used
syms fs xs ys
fs = xs.^2 + ys.^2 - 2*ys + 3*xs;


x = linspace(-20,20,100);
y = x';
z = f(x,y);
surf(x,y,z);


dfx = diff(fs,xs);
dfy = diff(fs,ys);

% converts symbolic functions to anonymous functions
gradx = matlabFunction(dfx);
grady = matlabFunction(dfy);

X0 = [5,3];
alpha = .5;
iter = 1;

for i = 1:20
    X0(1) = X0(1) - alpha * gradx(X0(1));
    X0(2) = X0(2) - alpha * grady(X0(2));
end


end