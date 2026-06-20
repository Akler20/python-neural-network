import numpy as np
import nnfs

from nnfs.datasets import spiral_data,vertical_data
nnfs.init()

class Layer_Dense:
    def __init__(self,input_no,neuron_no,lambda_weights,lambda_bias,l1_l2=1):
        self.input=input_no
        self.weights=0.01*np.random.randn(input_no,neuron_no)
        self.bias=np.zeros((1,neuron_no))
        self.lambda_weights=lambda_weights
        self.lambda_bias=lambda_bias
        self.l1_l2=l1_l2
    def forward(self, inputs):

        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.bias
        
   
    def backward(self,dvalues):
        self.L1_weights=self.weights.copy()
        self.L1_bias=self.bias.copy()
        self.L2_weights=self.weights.copy()
        self.L2_bias=self.bias.copy()
        if self.l1_l2==0:
            self.L1_weights[self.L1_weights<0]=-1
            self.L1_weights[self.L1_weights>0]=1
            self.L1_bias[self.L1_bias<0]=-1
            self.L1_bias[self.L1_bias>0]=1
            self.dweights = np.dot(self.inputs.T, dvalues)+self.lambda_weights * self.L1_weights
            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)+self.lambda_bias*self.L1_bias
        elif self.l1_l2==1:
            self.L2_weights=2*self.lambda_weights*self.L2_weights
            self.L2_bias=2*self.lambda_bias*self.L2_bias
            self.dweights = np.dot(self.inputs.T, dvalues)+self.L2_weights

            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)+self.L2_bias
        
        else:
            self.dweights = np.dot(self.inputs.T, dvalues)

            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)
class Activation_ReLU:
    def forward(self, inputs):

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

# Common loss class
class Loss:
 # Calculates the data and regularization losses
 # given model output and ground truth values
 def calculate(self, output, y):
  # Calculate sample losses
  sample_losses = self.forward(output, y)
  # Calculate mean loss
  data_loss = np.mean(sample_losses)
  # Return loss
  return data_loss
    
class Activation_Softmax:
    def forward(self,inputs):
        self.inputs=inputs
        self.inputs=self.inputs-np.max(self.inputs,axis=1,keepdims=True)
        inputs_exp=np.exp(self.inputs)
        self.output=inputs_exp/np.sum(inputs_exp,axis=1,keepdims=True)
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
      
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate=learning_rate
        self.decay=decay
        self.momentum=momentum
    
    def pre_update_params(self,epoch):
        
        self.learning_rate/=(1+epoch*self.decay)
    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.bias)
            weight_updates=layer.weight_momentums*self.momentum-layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates=layer.bias_momentums*self.momentum-layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.bias += bias_updates

class Optimizer_AdaGrad:

    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

    def pre_update_params(self, epoch):

        self.learning_rate /= (1 + epoch * self.decay)

    def update_params(self, layer):

        # Create cache arrays if they do not exist
        if not hasattr(layer, 'weightcache'):

            layer.weightcache = np.zeros_like(layer.weights)
            layer.biascache = np.zeros_like(layer.bias)

        # Accumulate squared gradients
        layer.weightcache += layer.dweights ** 2
        layer.biascache += layer.dbiases ** 2

        # Parameter update
        layer.weights += -self.learning_rate * layer.dweights / (
            np.sqrt(layer.weightcache) + self.epsilon
        )

        layer.bias += -self.learning_rate * layer.dbiases / (
            np.sqrt(layer.biascache )+ self.epsilon
        )
 

class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                           (1 - self.rho) * layer.dbiases**2
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
    def pre_update_params(self,epoch):
        self.iterations=epoch
        self.learning_rate/=(1+self.decay*self.iterations)
    def update_params(self, layer):
        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.bias)
            layer.bias_cache = np.zeros_like(layer.bias)
        layer.weight_cache=self.beta_2*layer.weight_cache+(1-self.beta_2)*layer.dweights**2
        layer.bias_cache=self.beta_2*layer.bias_cache+(1-self.beta_2)*layer.dbiases**2
        layer.weight_momentums=self.beta_1*layer.weight_momentums+(1-self.beta_1)*layer.dweights
        layer.bias_momentums=self.beta_1*layer.bias_momentums+(1-self.beta_1)*layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights+=-self.learning_rate*weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.bias+=-self.learning_rate*bias_momentums_corrected/(np.sqrt(bias_cache_corrected)+self.epsilon)
class Dropout:

    def forward(self, rate, layer):

        self.rate = rate

        # create mask
        self.drop = np.random.binomial(
            1,
            1 - self.rate,
        size=layer.output.shape
        )
        # apply mask
        layer.output = layer.output * self.drop
        # scale
        layer.output /= (1 - self.rate)

    def backward(self, dvalues):
    
        # same mask applied to gradients
        self.dinputs = dvalues * self.drop
        # scale gradients too
        self.dinputs /= (1 - self.rate)
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(
    2,
    100,
    lambda_weights=5e-4,
    lambda_bias=5e-4,
    l1_l2=3
)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(
    100,
    3,
    lambda_weights=5e-4,
    lambda_bias=5e-4,
    l1_l2=1
)
# Create Softmax classifier’s combined loss and activation
Dropout=Dropout()
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function
# takes the output of first dense layer here

Dropout.forward(rate=0.1,layer=dense1)
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

loss = loss_activation.forward(dense2.output, y)
print

# Let’s see output of the first few samples:
print(loss_activation.output[:5])
# Print loss value
print('loss:', loss)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
 y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
# Print accuracy
print('acc:', accuracy)
# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
Dropout.backward(activation1.dinputs)

dense1.backward(Dropout.dinputs)
# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
optimizer=Optimizer_Adam(learning_rate=0.02, decay=1e-5)
for i in range(10000):

    # Forward
    dense1.forward(X)
    Dropout.forward(rate=0.1,layer=dense1)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    accuracy = np.mean(predictions == y)
    if i%100==0:
        print(i, "loss:", loss, "acc:", accuracy)

    # Backward
    loss_activation.backward(loss_activation.output, y)
   
    
    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)
    Dropout.backward(activation1.dinputs)

    dense1.backward(Dropout.dinputs)

    # Update
  

    
    optimizer.pre_update_params(1)
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    
# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
 y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')