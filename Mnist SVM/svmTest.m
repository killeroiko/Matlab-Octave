%laod the trained model
load("svmModel.mat");

%load the test data
test_data = load('mnist_test.csv'); 

%pick the labels of the data
test_labels = test_data(:,1);

%pick the images of the data
test_images = test_data(:,2:end);

%normalize the images
test_images = test_images(:,:)/255;

%pick odd and even numbers apart
for i = 1:length(test_labels)/10
  if mod(test_labels(i),2) == 0
    test_digit_class(i) = 0;
    %create input data
    test_input(i,:) = test_images(i,:);
  elseif mod(test_labels(i),2) == 1
    test_digit_class(i) = 1;
    %create input data
    test_input(i,:) = test_images(i,:);
  end
end  

%create examples input
test_example = test_digit_class';

prediction = svmPredict(model,test_input);

Result = 0;

for i = 1:length(test_example)
  if prediction(i) == test_example(i)
    Result += 1;
   end
end

Accuracy = Result/length(test_example) * 100