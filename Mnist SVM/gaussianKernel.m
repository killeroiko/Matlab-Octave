function sim = gaussianKernel(x1, x2, sigma)

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

sim = 0;

magnitude = sum((x1-x2).^2);
sim = e^(-magnitude/(2*sigma^2));
    
end
