import random
from typing import Tuple

import optax
import jax.numpy as jnp
import jax
import numpy as np



def variational_circuit(x: jnp.ndarray, params: jnp.ndarray, n_layer: int) -> jnp.ndarray:
  # TODO: this has to be replaced with our circuit where x is fed into the encoder and params['w'] is fed into 
  x = jnp.dot(x, params['hidden'])
  x = jax.nn.relu(x)
  x = jnp.dot(x, params['output'])
    
  # TODO: like that
  #x_ = encoder(x)
  #x_ = weights(x)
    
  return x



def run_training(X, y, circuit:'Circuit', simulator:'Simulator', optimizer=None):
    
    n_layers = 2
    batch_size = 32
    epochs = 1_000
    
    def circuit_wrapper(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        circuit_fn = circuit.encode(x, params)
        
        simulator = simulate.run(circuit_fn)

        # TODO histogram to label

        # TODO return label
        pass

    
    def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        y_hat = circuit_wrapper(batch, params, circuit_fn)

        # TODO converter histogram_to_label
        # optax also provides a number of common loss functions.
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
    
    params, losses = fit(initial_params, optimizer)
    
    return params, losses