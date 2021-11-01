function memberships = FindClosestCentroids(X, centroids)

 %set the number of centers
 k = size(centroids, 1);

 %set the number of data points
 data_points = size(X, 1);

 %memberships will hold the cluster numbers for each example
 memberships = zeros(data_points, 1);

 %create a data point and cluster distance matrix
 distances = zeros(data_points, k);

 %compute the squared distance
 for i = 1:k
    
     %subtract centroid i from all data points
     diffs = bsxfun(@minus, X, centroids(i, :));
    
     %square the differences
     sqrdDiffs = diffs .^ 2;
    
     %take the sum of the squared differences
     distances(:, i) = sum(sqrdDiffs, 2);

 end

 %find the minimum distance value and set its index
 [minimum_values memberships] = min(distances, [], 2);

end

