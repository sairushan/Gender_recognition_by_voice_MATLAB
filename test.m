
load weights_final_89;

X = csvread("voice.csv");
y = (X(1:end, 21))';  % 1 * 3168
X = (X(1:end, 1:20))';  % 20 * 3168

X = normalize(X);

[A2,A1,Z1,Z2,A3,Z3] = fwd(X,W1,b1,W2,b2,W3,b3,y);
p = A3>0.5;
acc1 = mean(double(p == y)) * 100;
fprintf("Accuracy = %f \n",acc1);



function [A2,A1,Z1,Z2,A3,Z3] = fwd(X,W1,b1,W2,b2,W3,b3,y)
    m = size(X,2);
    Z1 = ((W1 * X) + b1);  % 4 * 891;
    A1 = tanh(Z1);

    Z2 = (W2 * A1) + b2; 
    A2 = tanh(Z2);

    Z3 = (W3 * A2) + b3; % 1 * 1;
    A3 = sigmoid(Z3);
end

function X = normalize(X)
    [n m] = size(X);
    mu = mean(X);
    X = bsxfun(@minus,X,mu);
    sigma = std(X);
    X = bsxfun(@rdivide,X,sigma);
    end
        
