function [output] = PCATest(X)

  %normalise the data
  X_mean = mean(X);
  X_tilde = X-X_mean;

  %calculate covariance matrix
  covariance_matrix = (X_tilde'*X_tilde)/(length(X)-1);

  %calculate eigenvectors and eigenvalues of covariance matrix
  [evec, eval] = eig(covariance_matrix);
  total = sum(sum(eval));
  eval = max(eval);

  %choose k eigenvectors
  explained_variance = 0.71;
  for i = 0:783
      test = sum(eval((784-i):784))/total;
      if test >= explained_variance
          k = i;
          break
      end
  end

  Vk = evec(:,((784-(k-1)):784));

  %project images onto reduced dimensionality eigenbasis
  output = X_tilde*Vk;
  
end