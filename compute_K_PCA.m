%% Neural Network computation of variance retained with reduction
%% by Jean Dessain - 28 May 2020
%% This file is a personal development from the CS229 Standford class exercise
%% It computes the variance retained per level of reduction based on the 
%% diagonal of the matrix Sigma

%% Setup the parameters you will use for this exercise
data_file = 'dataR_BGW.mat';      % choose the dataset for training and cross-validation
input_layer_size  = 400;  % Size of the matrix X
train_size = 5000;        % the training set can be smaller than the total size of the matrix X
output = 1;               % 1 = retained variance as function of K; 2 = loss from reduction and recovery (matrix with size = X
output_type = 2;          % 1 = matrix ; 2 = graph
K = 250;                  % Size of Z for the computation of the loss due to matrix reduction with PCA
                          % K is only used if output = 2
%% =========== Part 0: Initialization  ===================
% clear ; close all; clc
fprintf('Initializing ...\n')

%% Initialization
% clear ; close all; clc
                       
%% =========== Part 1: Loading and Visualizing Data =============
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

%% Define X for training and variance retained computation

X_train = X(1:train_size , :);      % define part of X used for training set

%% ================ Part 2: OPTIONAL - Randomly select 100 data points to display
% fprintf('Visualizing initial Data ...\n')
% sel = randperm(size(X_train, 1));
% sel = sel(1:100);
% displayData(X_train(sel, :));

%% ================ Part 3: Running PCA on initial data ================
 
X_norm = X_train;
[U, S] = pca(X_norm);
[U, S] = pca(X_norm);
kk = size(X_norm, 2);
Reduc_loss = zeros(kk, 2);
b = trace(S);
K_var = zeros(kk, 2);
a = 0;
for i = 1:kk;
  a = a + S(i, i);  
  K_var(i, :) = [i, a/b];
endfor

%% ================ Part 4: Computing the resteored X and the loss through reduction ================
Z_train = projectData(X_norm, U, K);
X_rec = recoverData(Z_train, U, K);   % recover X from Z 
Reduc_loss = X_train - X_rec;         % compute the loss through reduction and restoration

%% ================ Part 5: Output ================
if (output_type ==1)
  if (output==1)
    K_var
  else
    Reduc_loss
  endif
else
  if (output==1)
    figure;
    plot(K_var(:,1), K_var(:,2));
    xlabel ("K");
    ylabel ("Variance retained");
  else
    figure;
    displayData(X_train(sel, :));
    figure;
    displayData(X_rec(sel, :));
  endif
endif





