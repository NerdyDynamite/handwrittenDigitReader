function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    m = length(y); % number of training examples

    % Variables to return
    J = 0;
    grad = zeros(size(theta));

    h = sigmoid(X*theta);                                                       % probability function
    J = -(1/m)*(y'*log(h)+(1-y')*log(1-h))+(lambda/(2*m))*sum(theta(2:end).^2); % cost function for logistic regression

    grad = (1/m)*X'*(h-y);          % calculate the gradient
    temp = theta;
    temp(1) = 0;                    % do not add anything for j=0 (first feature is not part of the regularization procedure)
    grad = grad + (lambda/m)*temp;  % add the regularization term to each gradient value

    grad = grad(:); % ensure the gradient vector is a column vector

end
