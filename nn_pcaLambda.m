%% Neural Network with pca reduction
%% by Jean Dessain - 27 May 2020
%% This file is a personal development from the CS229 Standford class exercise
%% for neural network with matrix X reduction from n to K size
%% and look for best lambda of single hidden layer NN

%% =========== Part 0: Initialization  ===================
% clear ; close all; clc
fprintf('Initializing ...\n')

%% Initialization
% clear ; close all; clc

%% Setup the parameters you will use for this exercise
data_file = 'dataRe_BGW.mat';      % choose the dataset for training and cross-validation
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to JDE 11 as some images are not numbers   
train_size = 4000;        % the training set can be smaller than the total size of the matrix X
cv_size_init = 4001;    % the test set start at position test_size_init that can be bigger than 1 => reducing the test
Max_iternn = 100;                     % Number of iterations or training the neural network
K = 100;                  % size of the reduced Z matrix that replaces X of size input_layer_size                           
nb_iter_lambda = 12;      % number of tests to optimize lambda                       
                       % (note that we have mapped "0" to label 10)
L = zeros(nb_iter_lambda , 1);
J_train = zeros(nb_iter_lambda, 1);
P_train = zeros(nb_iter_lambda, 1);
J_test = zeros(nb_iter_lambda, 1);
P_test = zeros(nb_iter_lambda, 1);
Lambda_opt = zeros(nb_iter_lambda , 5);
                       
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%% A. Loading dataset from .mat file
load(data_file);
%% End version A.
 
%% OR  %%
 
%% B. loading dataset from an excel file
% pkg load io;
% pkg load windows;
% XLS = xlsread('Mat_excel_BW.xlsx');
% X = XLS(2:end, 2:401);
% y = XLS(2:end, 403);
%% End Version B.

m = size(X, 1);

if m < train_size                   % constraining the size of the training set
   train_size = m;
endif
if m < cv_size_init                   % constraining the size of the cross-validation set
   cv_size_init = m - 100;
endif
%% Define X for training and for test 

X_train = X(1:train_size , :);      % define part of X used for training set
y_train = y(1:train_size , 1);      % define part of y used for training set
m_train = size(X_train, 1);         % define size of X used for raining set

X_test = X(cv_size_init:end , :);   % define part of X for cross_validation, starting from initial point
y_test = y(cv_size_init:end , 1);   % define part of y for cross_validation
m_test = size(X_test, 1);             % define size of X used for cross_validation

%% ================ Part 2: Display selection ================
%% Randomly select 100 data points to display
 fprintf('Visualizing initial Data ...\n')
 sel = randperm(size(X_train, 1));
 sel = sel(1:100);

 displayData(X_train(sel, :));

%% ================ Part 3: Seraching for best K ================
%  In this part of the exercise, you will be starting to implment a two
 
X_norm = X_train;
[U, S] = pca(X_norm);
kk = size(X_norm, 2);
b = trace(S);
K_var = zeros(kk, 2);
a = 0;
for i = 1:kk;
  a = a + S(i, i);  
  K_var(i, :) = [i, a/b];
endfor
K_var
pause

X_norm = X_train;
[U, S] = pca(X_norm);
Z_train = projectData(X_norm, U, K);
Z_test = projectData(X_test, U, K);
 
sel = sel(1:100);
% displayData(Z_train(sel, :));
 
 
%% ================ Part 3: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(K, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Part 3: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', Max_iternn);

%  Testing different values of lambda
lambda = 0.01;
ind = 1;
max_iter = nb_iter_lambda(1,1);
for ind = 1:max_iter
% For ind = 1:nb_iter_lambda
  
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, Z_train, y_train, lambda);

  
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (K + 1)), hidden_layer_size, (K + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (K + 1))):end), num_labels, (hidden_layer_size + 1));

J = nnCostFunction(nn_params, K, hidden_layer_size, num_labels, Z_train, y_train, lambda);
pred = predict(Theta1, Theta2, Z_train);
J_train = J;
P_train = mean(double(pred == y_train)) * 100;

% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('Training Set Accuracy: %f\n', P_train(ind));
J = nnCostFunction(nn_params, K, hidden_layer_size, num_labels, Z_test, y_test, lambda);

pred_test = predict(Theta1, Theta2, Z_test);
J_test(ind, 1) = J;
P_test(ind, 1) = mean(double(pred_test == y_test)) * 100;

fprintf('Test Set Accuracy: %f\n', P_test(ind));
L(ind, 1) = lambda;


lambda = lambda * 2;

end


fprintf('\nResult : Error and adequacy per lambda... \n')
Lambda_opt = [L J_train J_test P_train P_test]
