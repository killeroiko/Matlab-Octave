function activation_value = GetRBFActivations(centers, betas, input)

 %subtract the input from all of the centers
 diffs = bsxfun(@minus, centers, input);
    
 %take the sum of squared Euclidean distance
 sqrdDists = sum(diffs .^ 2, 2);

 %apply the beta coefficient and take the negative exponent
 activation_value = exp(-betas .* sqrdDists);

end