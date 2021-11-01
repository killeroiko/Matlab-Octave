%load the training data
train_data = load('mnist_train.csv'); 

%pick the labels of the data
train_labels = train_data(:,1);

%pick the images of the data
train_images = train_data(:,2:end);

%normalize the images to create input ready data
train_images = train_images(:,:)/255;

%pick odd and even numbers apart
for i = 1:length(train_labels)/10
  if mod(train_labels(i),2) == 0
    train_digit_class(i) = 0;
    %create input data
    train_input(i,:) = train_images(i,:);
  elseif mod(train_labels(i),2) == 1
    train_digit_class(i) = 1;
    %create input data
    train_input(i,:) = train_images(i,:);
  end
end   

%create examples input
train_example = train_digit_class';

%train svm
C = 1;
model = svmTrain(train_input, train_example, C, @linearKernel);

%predict and mesure the accuracy of the training
prediction = svmPredict(model,train_input);

Result = 0;

for i = 1:length(train_example)
  if prediction(i) == train_example(i)
    Result += 1;
   end
end

Accuracy = Result/length(train_example) * 100

save svmModel.mat model;