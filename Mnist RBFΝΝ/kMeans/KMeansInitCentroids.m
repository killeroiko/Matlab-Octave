function centroids = KMeansInitCentroids(X, k)

 centroids = zeros(k, size(X, 2));

 %randomly reorder the indices of examples
 randidx = randperm(size(X, 1));

 %take the first k examples as centroids
 centroids = X(randidx(1:k), :);

end

