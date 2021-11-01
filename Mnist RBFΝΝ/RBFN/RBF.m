function [centers, betas, weights] = RBF(X_train, y_train, centers_per_category)
  
 %get the number of unique categories in the dataset
 unique_categories = size(unique(y_train), 1);
    
 %set the number of data points
 data_points = size(X_train, 1);
    
 centers = [];
 betas = [];    
    
 for (c = 1:unique_categories)
        
     %select the training vectors for category c
     Xc = X_train((y_train == c), :);
     
     %select random initial centroids
     init_centroids = KMeansInitCentroids(X_train,centers_per_category);
        
     %run k-means clustering with at most 100 iterations
     [centroids_c, memberships_c] = KMeans(Xc, init_centroids, 100);    
        
     %remove any empty clusters
     to_remove = [];
        
     for (i = 1:size(centroids_c, 1))
       
         %if this centroid has no members mark it for removal
         if (sum(memberships_c == i) == 0)        
             to_remove = [to_remove; i];
         end
     end
        
     %if there were empty clusters
     if (~isempty(to_remove))
       
         %remove the centroids of the empty clusters
         centroids_c(to_remove, :) = [];
            
         %reassign the memberships
         memberships_c = FindClosestCentroids(Xc, centroids_c);
     end
        
     %compute betas for all the clusters
     betas_c = ComputeRBFBetas(Xc, centroids_c, memberships_c);
        
     %add the centroids and their beta values to the network
     centers = [centers; centroids_c];
     betas = [betas; betas_c];
 end

 %get the final number of RBF neurons
 number_of_neurons = size(centers, 1);

 %compute the RBF neuron activations for all training examples
 X_activation = zeros(data_points, number_of_neurons);

 for (i = 1:data_points)
       
     input = X_train(i, :);
       
     %get the activation for all RBF neurons for this input
     activation_value = GetRBFActivations(centers, betas, input);
       
     %store the activation values each example
     X_activation(i, :) = activation_value';
 end

 %add a column of 1s for the bias term
 X_activation = [ones(data_points, 1), X_activation];

 %create a weights matrix
 weights = zeros(number_of_neurons + 1, unique_categories);

 for (c = 1:unique_categories)

     %make the y values binary. 1 for category c and 0 for all the other
     y_c = (y_train == c);

     %use the normal equations to solve for optimal theta
     weights(:, c) = pinv(X_activation' * X_activation) * X_activation' * y_c;
 end
   
end

