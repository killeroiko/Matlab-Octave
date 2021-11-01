%add the subdirectories to the path
addpath('kMeans');
addpath('RBFN'); 

%laod the trained network
load('TrainedRBF.mat');

%load the training data
test_data = load('mnist_test.csv');

%create the labels array
test_labels = test_data(:,1); 

%create the images array
test_images = test_data(:,2:785); 

%normalize the data
test_images = test_images(:,:)/255;

%create the input for the neural network with pca
input_images = PCATest(test_images);

%set the number of data points
data_points = size(input_images, 1);

%this allows the index of the output node to equal its category 
input_labels = test_labels + 1;

%mesure the accuracy 
correct = 0;

for (i = 1:data_points)
  
    %compute the scores
    scores = EvaluateRBFN(centers, betas, weights, input_images(i, :));
        
	  [max_score, category] = max(scores);
	
    %validate the result
    if (category == input_labels(i))
        correct += 1;    
    end
    
end

%display the accuracy
accuracy = correct / data_points * 100