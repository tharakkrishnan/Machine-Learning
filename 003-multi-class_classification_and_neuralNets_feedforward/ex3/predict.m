function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add the bias elements in input need to be added

X = resize(X,size(X, 1),size(X, 2)+1);
X = circshift(X,[0,1]);
X(:,1) = 1;

% X has size 5000 x 401
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

%Compute layer 1 size 5000 x 25
A1 = sigmoid(X * Theta1');

A1 = resize(A1,size(A1, 1),size(A1, 2)+1);
A1 = circshift(A1,[0,1]);
A1(:,1) = 1;

%Compute layer 2 size 5000 x 10
A2 = sigmoid(A1 * Theta2');
[v,p] = max(A2'); 
p=p';

% =========================================================================


end
