function centroids = ComputeCentroids(X, prev_centroids, memberships, k)

 %X samples and dimensions
 [samples dimensions] = size(X);

 centroids = zeros(k, dimensions);

 for (i = 1:k)
   
     %if no points are assigned to the centroid, don't move it
     if (~any(memberships == i))
         centroids(i, :) = prev_centroids(i, :);
         
     %otherwise, compute the cluster's new centroid
     else
     
         %select the data points assigned to centroid k
         points = X((memberships == i), :);

         %compute the new centroid as the mean of the data points
         centroids(i, :) = mean(points);    
     end
 end

end

