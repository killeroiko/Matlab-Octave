%add the subdirectories to the path
addpath('kMeans');
addpath('RBFN'); 

%load the training data
train_data = load('mnist_train.csv');

%create the labels array
train_labels = train_data(:,1); 

%create the images array
train_images = train_data(:,2:785); 

%normalize the data
train_images = train_images(:,:)/255;

%create the input for the neural network with pca
input_images = PCA(train_images);

%this allows the index of the output node to equal its category 
input_labels = train_labels + 1;

%set the number of data points
data_points = size(input_images, 1);

%train the RBF using 10 centers per category
[centers, betas, weights] = RBF(input_images, input_labels, 10);

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

%save the trained network
save('TrainedRBF.mat', 'centers', 'betas', 'weights');
