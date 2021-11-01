function label = gnb(X,Y,Xnew)
  
[C,~,Y] = unique(Y);
n_labels = length(C);
[P, N] = size(Xnew);
label = zeros(P,1);
for inst = 1:P
    probability = histc(Y,1:max(Y))/length(Y);
    for j = 1:n_labels
        data = X(Y==j, :);
        standard_deviation = std(data);
        average = mean(data);
        for i = 1:N
            gauss = 1/(standard_deviation(i)*sqrt(2*pi))*exp(-1/2*((Xnew(inst,i)-average(i))/standard_deviation(i))^2);
            probability(j) = probability(j)*gauss;
        end
    end
    
    [~,I] = sort(probability,'descend');
    label(inst) = I(1);
end
label = C(label);
endfunction
