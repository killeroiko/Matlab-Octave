%load the training data

train_data = load ('mnist_train.csv');

%create the labels array

train_labels = train_data(:,1); 

%create the images array

train_images = train_data(:,2:785); 

%normalize the data
%create the input for the neural network

train_images = train_images(:,:)/255; 

%create the predetermined output of the neural network
%the data need to be in categorical array form

for i=1:length(train_labels)
  if train_labels(i) == 0;
    correct_Output(i,:) = [1,0,0,0,0,0,0,0,0,0];
  elseif train_labels(i) == 1;
    correct_Output(i,:) = [0,1,0,0,0,0,0,0,0,0];
  elseif train_labels(i) == 2;
    correct_Output(i,:) = [0,0,1,0,0,0,0,0,0,0];
  elseif train_labels(i) == 3;
    correct_Output(i,:) = [0,0,0,1,0,0,0,0,0,0];
  elseif train_labels(i) == 4;
    correct_Output(i,:) = [0,0,0,0,1,0,0,0,0,0];
  elseif train_labels(i) == 5;
    correct_Output(i,:) = [0,0,0,0,0,1,0,0,0,0];
  elseif train_labels(i) == 6;
    correct_Output(i,:) = [0,0,0,0,0,0,1,0,0,0];
  elseif train_labels(i) == 7;
    correct_Output(i,:) = [0,0,0,0,0,0,0,1,0,0];
  elseif train_labels(i) == 8;
    correct_Output(i,:) = [0,0,0,0,0,0,0,0,1,0];
  elseif train_labels(i) == 9;
    correct_Output(i,:) = [0,0,0,0,0,0,0,0,0,1];  
  end
end

%create the weight arrays and batch size

Batch = length(train_images);

w1 = randn(64,784)*sqrt(2/784);
w2 = randn(64,64)*sqrt(2/64);
w3 = randn(64,64)*sqrt(2/64);
w4 = randn(10,64)*sqrt(2/64);

%train the network for 5 epochs

for epoch = 1:5
  [w1, w2, w3, w4] = DeepLearning(w1, w2, w3, w4, train_images, Batch, correct_Output);
endfor

%save the trained network as .mat format

save('DeepNeuralNetwork.mat')
                  