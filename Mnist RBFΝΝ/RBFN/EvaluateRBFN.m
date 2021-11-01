function output = EvaluateRBFN(centers, betas, weights, input)
    
 %compute the activations for each RBF neuron for this input
 activation = GetRBFActivations(centers, betas, input);
    
 %add a 1 to the beginning of the activations vector for the bias term
 activation = [1; activation];
    
 output = weights' * activation;
        
end