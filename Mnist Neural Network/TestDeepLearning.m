%load the trained neural network

load('DeepNeuralNetwork.mat');

%load the testing data

test_data = load('mnist_test.csv');

%create the labels array

test_labels = test_data(:,1);

%creat the images array

test_images = test_data(:,2:785);

%normalize the data

test_images = test_images(:,:)/255;

%begin evaluation

e = 0.1;
success = 0;

for i=1:length(test_images)

  %create the input image

  input_Image = test_images(i,:)';

  %run the image through the network
               
  input_of_hidden_layer1 = w1*input_Image;
  output_of_hidden_layer1 = ReLU(input_of_hidden_layer1);

  input_of_hidden_layer2 = w2*output_of_hidden_layer1;
  output_of_hidden_layer2 = ReLU(input_of_hidden_layer2);

  input_of_hidden_layer3 = w3*output_of_hidden_layer2;
  output_of_hidden_layer3 = ReLU(input_of_hidden_layer3);

  input_of_output_node = w4*output_of_hidden_layer3;
  final_output = Softmax(input_of_output_node);
  
  compare = 0;
  output_label = 0;
  for k = 1:10
    if final_output(k) > compare
        output_label = k-1;
        compare = final_output(k);
    end
  end
  
  if test_labels(i) == output_label
    success = success + 1;
  end
  
end

accuracy = (success/length(test_images))*100
