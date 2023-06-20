# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)

def fun_r(x, lamda_i, y_i, y_x, y_xx):
    return y_xx + lamda_i*y_i

N_b = 200
N_r = 100
yb = 0

tf.random.set_seed(0)

lb = tf.constant(0, dtype=DTYPE)
# Upper bounds
ub = tf.constant(1, dtype=DTYPE)

X_b = tf.linspace(0,1,2)
y_b = yb*tf.ones((len(X_b),1))

X_r = tf.random.uniform((N_r,1), lb, ub, dtype=DTYPE)

X_data = [X_b]
y_data = [y_b]

def init_model(num_hidden_layers=8, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
                lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)
    # act_list = [tf.keras.activations.get('tanh'), tf.keras.activations.get('relu'), tf.keras.activations.get('mish'), tf.keras.activations.get('silu')]
    # Append hidden layers
    for _ in range(num_hidden_layers):
        # act_func = random.choice(act_list)
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('silu'),
            kernel_initializer='glorot_normal'))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    
    return model

def get_r(model, X_r, lamda_i, y_i):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        x = X_r[:, 0:1]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(x)

        # Determine residual 
        y = model(x)

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        y_x = tape.gradient(y, x)
            
    # u_t = tape.gradient(u, t)
    y_xx = tape.gradient(y_x, x)

    del tape

    return fun_r(x, lamda_i, y_i, y_x, y_xx)

def compute_loss(model, X_r, X_data, y_data, lamda_i, y_i):
    
    # Compute phi^r
    r = get_r(model, X_r, lamda_i, y_i)
    y_r = tf.reduce_mean(tf.square(r))
    
    # Initialize loss
    loss = y_r
    
    # Add phi^0 and phi^b to the loss
    for i in range(len(X_data)):
        y_pred = model(X_data[i])
        loss += tf.reduce_mean(tf.square(y_data[i] - y_pred))
    
    return loss

def get_grad(model, X_r, X_data, y_data, lamda_i, y_i):
    
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, X_r, X_data, y_data, lamda_i, y_i)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g

# Initialize model aka u_\theta
model = init_model(4, 10)

# We choose a piecewise decay of the learning rate, i.e., the
# step size in the gradient descent type algorithm
# the first 1000 steps use a learning rate of 0.01
# from 1000 - 3000: learning rate = 0.001
# from 3000 onwards: learning rate = 0.0005

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])

# Choose the optimizer
optim = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

from time import time

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step(lamda_i, phi_i):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad(model, X_r, X_data, y_data, lamda_i, phi_i)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss

def infty_norm(x):
    return np.max(np.abs(x))

# Number of training epochs
N = 5000

lamda0 = 10.0
def base_model(x):
    return x*(x-1)
y0 = base_model(X_r)
arr_lamda = [lamda0]
arr_phi = [y0]

# Start timer
for j in range(10):
    t0 = time()
    hist = []
    for i in range(N+1):
        
        loss = train_step(arr_lamda[-1], arr_phi[-1])
        
        # Append current loss to hist
        hist.append(loss.numpy())
        
        # Output current loss after 50 iterates
        if i%50 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(i,loss))
        if i > 100:
            if (np.abs(np.diff(np.array(hist)))[-100:] < 1e-5).all():
                print('stop iteration for j=',j,' is ',i)
                break
    if loss < 5e-2:
        phi_est = model(X_r)
        norm_phi = infty_norm(phi_est)
        arr_phi.append(phi_est/norm_phi)
        arr_lamda.append(arr_lamda[-1]/norm_phi)
        arr_lamda[-1] = arr_lamda[-1].astype('float32')
        print('lamda-', j,': ', arr_lamda[-1])
        print('norm phi: ', norm_phi)
    else:
        print('failed')
        break
            
    # Print computation time
    print('\nComputation time: {} seconds'.format(time()-t0))