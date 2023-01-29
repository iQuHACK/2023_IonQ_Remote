import random
from typing import Tuple

import optax
import jax.numpy as jnp
import jax
import numpy as np

from utils.utils import simulate, histogram_to_label

from sklearn.model_selection import train_test_split

def variational_circuit(x: jnp.ndarray, params: jnp.ndarray, n_layer: int) -> jnp.ndarray:
  # TODO: this has to be replaced with our circuit where x is fed into the encoder and params['w'] is fed into 
  x = jnp.dot(x, params['hidden'])
  x = jax.nn.relu(x)
  x = jnp.dot(x, params['output'])
    
  # TODO: like that
  #x_ = encoder(x)
  #x_ = weights(x)
    
  return x



def run_training(X, y, circuit:'Circuit', backend='qiskit', optimizer=None):
    
    n_layers = 2
    batch_size = 32
    epochs = 1_000
    
    def circuit_wrapper(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        circuit_fn = circuit.encode(x, params)
        
        histogram = simulate(circuit_fn, backend_=backend)
        
        return = histogram_to_label(histogram)

    
    def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        predictions = circuit_wrapper(batch, params, circuit_fn)
        
        y_pred = jax.nn.one_hot(y % 2, 2).astype(jnp.float32).reshape(len(labels), 2)
        loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

        return loss_value.mean()

    def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, batch, labels):
            loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for i in range(epochs):

            batch_index = np.random.randint(0, len(y), (batch_size,))
            x_train_batch = X[batch_index]
            y_train_batch = y[batch_index]


            params, opt_state, loss_value = step(params, opt_state, x_train_batch, y_train_batch)
            losses.append(loss_value)
            if i % 100 == 0:
                print(f'step {i}, loss: {loss_value}')

            return params, losses

    
    initial_params = {
        'w': jax.random.normal(shape=[16, n_layers], key=jax.random.PRNGKey(0)),
    }
    
    if optimizer is None:
        optimizer = optax.adam(learning_rate=1e-2)
    
    optimal_params, losses = fit(initial_params, optimizer)
    
    # TODO return optimal circuit
    
    return optimal_circuit, losses


# TODO load data

# TODO train test split

X = np.load('data/images.npy')
y = np.load('data/labels.npy')
y_reshape = jax.nn.one_hot(y % 2, 2).astype(jnp.float32).reshape(2000, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.33, random_state=42)


circuit_weights, losses = run_training(X_train, y_train, None)

