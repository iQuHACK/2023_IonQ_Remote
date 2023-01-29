import logging
import sys

from jax.config import config

config.update("jax_enable_x64", True)
import jax
import optax
from jax import numpy as jnp


import numpy as np

class Classifier():
    
    def __init__(self, debug = True):
        
        self.weights = None
        self.bias = None

        #self.state_preparation = state_preparation
        #self.layer = layer
        #self.n_layers = n_layers
        #self.min_wires = min_wires
        #self.n_wires = min_wires
        #self.batch_size = batch_size
        #self.epochs = epochs

        self.optimizer = optax.adam(learning_rate=0.05)

        self.debug = debug
        self.is_trained = False
    
    def fit(X, y):
        # Make features and labels differentiable
        x_train = np.array(X, requires_grad=False)
        y_train = np.zeros(shape=(len(y), self.n_wires), requires_grad=False)

        # Transform labels to encoding that matches the quantum computer with respective number of wires
        # TODO

        # Initialise weights and bias if not trained before, classifier can be retrained
        if self.weights is None:
            self.weights = 0.01 * jnp.random.randn(self.n_layers, self.n_wires, 3, requires_grad=True)
        if self.bias is None:
            self.bias = jnp.zeros(self.n_wires, requires_grad=True)

        # Iterate for some epochs and optimize the variational model
        n_train_samples = len(x_train)

        params = {'w': jnp.asarray(self.weights), 'b': jnp.asarray(self.bias)}
        opt_state = self.optimizer.init(params)

        def cost_(p_, x_, y_):
            predictions = [self._variational_classifier(p_['w'], feat_, p_['b']) for feat_ in x_]
            return self.cost_fn(y_, predictions)

        for i_ in range(self.epochs):
            # Create a random batch of samples
            batch_index = np.random.randint(0, n_train_samples, (self.batch_size,))
            x_train_batch = x_train[batch_index]
            y_train_batch = y_train[batch_index]

            # Optimize weights and bias
            if self.use_jax:
                cost, grad_circuit = jax.value_and_grad(lambda p: cost_(p, x_train_batch, y_train_batch))(params)
                updates, opt_state = self.optimizer.update(grad_circuit, opt_state)
                params = optax.apply_updates(params, updates)
                self.weights, self.bias = params['w'], params['b']

            if self.debug:
                sys.stdout.write(f'\rVQE -> Epoch:{i_}')
                sys.stdout.flush()

        sys.stdout.write('\n')

        return self

    
    def predict(self):
        # TODO call the model
        # TODO return 1 or 0
        pass
    
    def score(self):
        pass



