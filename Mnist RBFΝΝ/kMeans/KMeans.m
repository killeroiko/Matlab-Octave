function [centroids, memberships] = KMeans(X, initial_centroids, max_iters)

 %get the number of initial centroids
 k = size(initial_centroids, 1);

 centroids = initial_centroids;
 previous_centroids = centroids;

 %run K-Means
 for (i = 1:max_iters)
    
     %assign each example in X to the closest centroid
     memberships = FindClosestCentroids(X, centroids);
        
     %compute new centroids
     centroids = ComputeCentroids(X, centroids, memberships, k);
    
     %check if the centroids haven't changed since last iteration
     if (previous_centroids == centroids)
         break;
     end

     %update the previous centroids
     previous_centroids = centroids;
 end

end

