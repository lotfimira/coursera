function [error_test] = computeTestError(X, y, Xtest, ytest, lambda)

m = size(X, 1);
m_test = size(Xtest, 1);

Xtrain = [ones(m, 1) X];
ytrain = y;
[theta] = trainLinearReg(Xtrain, ytrain, lambda);

Xtest_prep = [ones(m_test, 1) Xtest];
[J, grad] = linearRegCostFunction(Xtest_prep, ytest, theta, 0);
error_test = J;

end