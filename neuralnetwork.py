class Network(object):
        #number_of_neurons_by_layer is a list containing number of neurons in each layer
        def __init__(self, number_of_neurons_by_layer):
            self.num_layers = len(number_of_neurons_by_layer)
            self.number_of_neurons_by_layer = number_of_neurons_by_layer

            #biases of every neuron in this array
            self.biases = [np.random.rand(y,1) for y in number_of_neurons_by_layer[1:]]
            
            self.weights = [np.random.rand(y,x) for x, y in zip(number_of_neurons_by_layer[:-1], number_of_neurons_by_layer[1:])]

        #biases to be initialized by random number generation

        #z can be a vector, and then exp will be applied on every element of vector
        def sigmoid(z):
            return 1.0/(1.0 + np.exp(-z))

        # Applies the equation activation = sigmoid(weight*activation + biases)
        def feedforward(self, a):
            """Return the output of the network if "a" is input"""
            for b,w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w,a) + b)
            return a

        #What inputs should go here
        #This is SGD = Stochastic Gradient Descent
        #On what is it applied
        
        def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
            """Training data is a list of tuples "(x,y)" representing the training inputs and desired outputs"""
            if test_data:
                n_test = len(test_data)
            n = len(training_data)
            for j in range(epochs):
                #randomly shuffle training data
                random.shuffle(training_data)
                #create mini batches of size of mini_batch size from training data
                mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

                #update for every mini_batch
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print("Epoch {0}: {1} / {2}").format(j, self.evaluate(test_data), n_test)
                else:
                    print("Epoch {0} complete").format(j)

        def update_mini_batch(self, mini_batch, eta):
            """Update the network's weights and biases by applying gradient
            descent using backpropagation to a single mini batch.
            The "mini batch" is a list of tuples "(x,y)" and "eta" is the learning rate."""

            #.shape returns size of each dimension 
            nabla_b = [np.zeros(b.shape) for b in self.biases] 
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            #x is input and y is output of a single sample
            for x,y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]

            self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weight, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

        def backprop(self, x, y):
            """Return a tuple "(nable_b, nabla_w)" representing the gradient of the cost function C_x.
            "nabla_b" and "nabla_w" are layer by layer lists of numpy arrays, similar
            to "self.biases" and "self.weights"."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]

             
            for b,w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)

        def evaluate(self, test_data):
            test_results = 
