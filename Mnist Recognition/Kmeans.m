function [means,Nmeans,membership] = Kmeans(X,K,maxerr)
[Ndata, dims] = size(X);
dist = zeros(1,K);
means = zeros(K,dims);
if (nargout > 2)
    membership = zeros(Ndata,1);
end
% Initial prototype assignment (arbitrary)
means(1,:) = median(X(:,1));
f=1;
for i=2:K
   d = X - mean(means);
   maxD = max(min(X'));
   means(i,:) = mean(means(i-1,:)) + f*maxD;
   f= f*(-1);
end
cmp = 1 + maxerr;
while (cmp > maxerr)
   % Sums (class) and data counters (Nclass) initialization
   class = zeros(K,dims);
   Nclass = zeros(K,1);
   % Groups each elements to the nearest prototype
   for i=1:Ndata
      for j=1:K
         % Euclidean distance from data to each prototype
         dist(j) = norm(X(i,:)-means(j,:))^2;
      end
      % Find indices of minimum distance
      index_min = find(~(dist-min(dist)));
      % If there are multiple min distances, decide randomly
      index_min = index_min(ceil(length(index_min)*rand));
      if (nargout > 2)
          membership(i) = index_min;
      end
      class(index_min,:) = class(index_min,:) + X(i,:);
      Nclass(index_min) = Nclass(index_min) + 1;
   end
   for i=1:K
      class(i,:) = class(i,:) / Nclass(i);
   end
   % Compare results with previous iteration
   cmp = 0;
   for i=1:K
      cmp = norm(class(i,:)-means(i,:)); 
   end
   % Prototype update
   means = class;
end
Nmeans = Nclass;

