%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: One-vs-All Training ============

% subdivide the data set into one for training, and one for testing
trainingSize = 4950; % number of examples to use for training
testingSize = m - trainingSize; % number of examples to use for testing

% Training sets. To be used for one-vs-all logistic regression training
trainingX = zeros(trainingSize,input_layer_size);
trainingY = zeros(trainingSize,1);

% Testing sets. To be used in part 3
testingX = zeros(testingSize,input_layer_size);
testingY = zeros(testingSize,1); 

rp = randperm(m);

for i=1:trainingSize
    trainingX(i,:) = X(rp(i),:);
    trainingY(i) = y(rp(i));
end

for j=(trainingSize+1):m
    testingX(j-trainingSize,:) = X(rp(j),:);
    testingY(j-trainingSize) = y(rp(j));  
end

% carry out the training
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(trainingX, trainingY, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, testingX);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == testingY)) * 100);

m = size(testingX, 1); % ensure you change size to that of the testing matrix X

rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(testingX(rp(i), :));

    pred = predictOneVsAll(all_theta, testingX(rp(i),:));
    fprintf('\nOne-vs-all Logistic Regression Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

