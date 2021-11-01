function betas = ComputeRBFBetas(X, centroids, memberships)

 %get the number of neurons
 number_of_neurons = size(centroids, 1);

 %compute sigma for each cluster
 sigmas = zeros(number_of_neurons, 1);
    
 for (i = 1:number_of_neurons)
     
     %select the next cluster centroid
     center = centroids(i, :);

     %select all of the members of this cluster
     members = X((memberships == i), :);

     %subtract the center vector from each of the member vectors
     differences = bsxfun(@minus, members, center);
        
     %take the sum of the squared differences
     sqrdDiffs = sum(differences .^ 2, 2);
        
     %take the square root to get the Euclidean distance
     distances = sqrt(sqrdDiffs);

     %compute the average Euclidean distance and use this as sigma
     sigmas(i, :) = mean(distances);
 end

 %verify no sigmas are 0
 if (any(sigmas == 0))
    error('One of the sigma values is zero!');
 end
    
 %compute the beta values from the sigmas
 betas = 1 ./ (2 .* sigmas .^ 2);
    
end