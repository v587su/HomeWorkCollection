#  Why does the standard cross entropy loss usually converges faster than the symmetrical formulation 

1. The symmetrical formulation like MSE needs to use Sigmoid as its activation function. And the sigmoid function is slow on the speed of convergence
2. According to the formulation, at the beginning of the learning the derivative of Cross Entropy Loss is mainly based on how far off our prediction was from the ground truth, which means the magnitude of derivative will be big.
