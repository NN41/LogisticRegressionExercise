# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# %% Generate 2D data with classes, and create train and test set

X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=0, cluster_std=1)
plt.scatter(X[:,0], X[:,1], c=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %% Define the logistic sigmoid function

def logistic_sigmoid(x):
    x_clipped = np.clip(np.array(x), -10**2.5, 10**2.5)
    return 1 / (1 + np.exp(-x_clipped))

x = np.linspace(-10,10,100)    
plt.plot(x, logistic_sigmoid(x))

# %% Define the prediction function and make a prediction using the initialized weights

weights = np.ones((X.shape[1], 1))
bias = 0.

def predict_y(X, weights, bias):
    return logistic_sigmoid(X @ weights + bias)

plt.scatter(X[:,0], X[:,1], c=predict_y(X, weights, bias))

# %% Compute binary cross entropy of prediction using the initialized weights

y_predicted = predict_y(X, weights, bias).flatten()
y_pred = np.where(y_predicted < 1e-10, 1e-10, y_predicted)
binary_cross_entropy = - np.average(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

# %% Compute the gradient and show that it is correct

yy = y[0]
xx = X[0]
weights = np.ones((X.shape[1], 1))
bias = 0.
epsilon = 1e-3
lambda_regu = 1

def bce(X, y, weights, bias): # binary cross entropy
    y_hat = predict_y(X, weights, bias).flatten()
    tolerance = 1e-10
    y_hat_clipped = np.clip(y_hat, tolerance, 1-tolerance)
    binary_cross_entropy = -np.average(y * np.log(y_hat_clipped) + (1-y) * np.log(1-y_hat_clipped))
    return float(binary_cross_entropy)

def criterion(X, y, weights, bias, lambda_regu):
    binary_cross_entropy = bce(X, y, weights, bias)
    regularization_term = lambda_regu * np.sum(weights * weights)
    return float(binary_cross_entropy + regularization_term)

S = xx @ weights + bias
deriv = -logistic_sigmoid(-S) * yy + logistic_sigmoid(S) * (1-yy)
grad2 = np.concatenate(((deriv * xx + 2 * lambda_regu * weights.T).flatten(), deriv))

print(f'gradients with L2 regu \n\t= {grad2}')
print(f'criterion = {criterion(xx,yy,weights,bias,lambda_regu)}')
print('\nNumerical check (should be roughly zero)')
print((criterion(xx,yy,weights+np.array([[epsilon],[0]]),bias,lambda_regu)-criterion(xx,yy,weights,bias,lambda_regu))/epsilon - grad2[0])
print((criterion(xx,yy,weights+np.array([[0],[epsilon]]),bias,lambda_regu)-criterion(xx,yy,weights,bias,lambda_regu))/epsilon - grad2[1])
print((criterion(xx,yy,weights,bias+epsilon,lambda_regu)-criterion(xx,yy,weights,bias,lambda_regu))/epsilon - grad2[2])

# %% Run a single gradient descent step using all the data (without regularization)

S = X @ weights + bias
deriv = (-y) * logistic_sigmoid(-S).flatten() + (1-y) * logistic_sigmoid(S).flatten()
grad = np.array([np.average(X[:,0] * deriv), np.average(X[:,1] * deriv), np.average(deriv)])

learning_rate = 1e-2
weights_new = weights - learning_rate * grad[:2].reshape((X.shape[1],-1))
bias_new = bias - learning_rate * grad[-1]

### PART 3 ###
# %%

def compute_gradient(X, y, weights, bias, lambda_regu): # uses L2 regularization
    S = X @ weights + bias
    deriv = (-y) * logistic_sigmoid(-S).flatten() + (1-y) * logistic_sigmoid(S).flatten()
    grad = np.array([
        np.average(X[:,0] * deriv) + 2 * lambda_regu * weights.flatten()[0], 
        np.average(X[:,1] * deriv) + 2 * lambda_regu * weights.flatten()[1], 
        np.average(deriv)
        ])
    return grad

def compute_new_params(grads, weights, bias, learning_rate):
    weights_new = weights - learning_rate * grads[:2].reshape((2,-1))
    bias_new = bias - learning_rate * grads[-1]
    return weights_new, bias_new

def training_run(X_train, y_train, weights, bias, epochs, learning_rate, lambda_regu):

    loss_list = []
    objective_list = []
    loss_test_list = []
    grads_list = []
    w1_list = []
    w2_list = []
    bias_list = []

    epoch = 0
    while epoch < epochs:
        loss = bce(X_train, y_train, weights, bias)
        objective = criterion(X_train, y_train, weights, bias, lambda_regu)
        loss_test = bce(X_test, y_test, weights, bias)
        grads = compute_gradient(X_train, y_train, weights, bias, lambda_regu)
        weights, bias = compute_new_params(grads, weights, bias, learning_rate)

        loss_list.append(loss)
        objective_list.append(objective)
        loss_test_list.append(loss_test)
        grads_list.append(grads)
        w1_list.append(weights[0])
        w2_list.append(weights[1])
        bias_list.append(bias)
        epoch += 1

    return loss_list, objective_list, loss_test_list, grads_list, w1_list, w2_list, bias_list

weights = np.ones((X.shape[1], 1))
bias = -10
epochs = 10000
learning_rate = 0.1
lambda_regu = 1

loss_list, objective_list, loss_test_list, grads_list, w1_list, w2_list, bias_list = training_run(
    X_train, y_train, weights, bias, epochs, learning_rate, lambda_regu
)

weights = np.array([w1_list[-1],w2_list[-1]])
bias = np.array(bias_list[-1])

print(f'final training error = {loss_list[-1]}')
print(f'final test error = {loss_test_list[-1]}')

fig, axes = plt.subplots(2, 2, figsize=(8,8))
axes = axes.flatten()
axes[0].plot(loss_list, label='training loss')
axes[0].plot(loss_test_list, label='test loss')
axes[0].plot(objective_list, label='objective')
axes[0].legend()

axes[1].plot(w1_list, label='w1')
axes[1].plot(np.array(grads_list)[:,0], label='deriv_w1')
axes[1].legend()

axes[2].plot(w2_list, label='w2')
axes[2].plot(np.array(grads_list)[:,1], label='deriv_w2')
axes[2].legend()

axes[3].plot(bias_list, label='bias')
axes[3].plot(np.array(grads_list)[:,2], label='deriv_bias')
axes[3].legend()

## Note how for make_blobs(n_samples=300, centers=2, n_features=2, random_state=0, cluster_std=1)
## the weights converge way quicker than the bias, which needs >1 order of magnitude more to converge
## This is because we are not taking into account the second order when applying a gradient step

## A few notes about regularization.
## (1) If your lambda is very high, then you need to take small steps, otherwise you will overshoot
## and the gradients will start spiraling out of control.
## (2) Since high regu pulls the weights to zero, this means that variations in x1 and x2 will lead
## to only small variations in the prediction, hence the prediction function is way less steep.
## But note that The decision boundary doesn't change, since the decision boundary is chosen
## such that the 0.5-level categorizes the highest number of correct data points. The weights
## only influence steepness of the prediction curve around the 0.5-level, but that doesn't
## matter for the classification. so ACCURACY doesn't change.
## (3) If we look at BCE, however, we actually measure how far the prediction is away from the
## actual class, in which case the regularization does severely degrade the performance.

# %% Plot decision boundary. Note how we can explicitly calculate the decision boundary by setting
# the prediction function equal to the desired level. It's obvious that it's linear, since we take
# an affine transformation of the input.

N_grid = 100
x1 = np.linspace(X[:,0].min(), X[:,0].max(), N_grid)
x2 = np.linspace(X[:,1].min(), X[:,1].max(), N_grid)
xx1, xx2 = np.meshgrid(x1,x2)
grid_points = np.vstack((xx1.flatten(), xx2.flatten())).T
grid_values = predict_y(grid_points, weights, bias)

def inverse_logistic_sigmoid(y):
    return -np.log(1/y-1)

def decision_boundary(x1, weights, bias, level):
    w1, w2 = weights.flatten()
    return (inverse_logistic_sigmoid(level)-bias-x1*w1)/w2

plt.scatter(grid_points[:,0], grid_points[:,1], c=grid_values, alpha=0.2)
plt.plot(x1, decision_boundary(x1, weights, bias, level=0.5), c='black')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.title('Prediction function with test set and with 0.5-level decision boundary')


# %%
# PART 4
# We see that we should not choose a learning rate that is too high, since we might keep overshooting
# the optimum instead of converging. Also not too low, because then convergence is super slow.

epochs = 100
learning_rates = [1000,1,0.01]

fig, axes = plt.subplots(2,1,figsize=(5,10))
axes = axes.flatten()

for lr in learning_rates:

    weights = np.ones((X.shape[1], 1))
    bias = 0.

    loss_list, loss_test_list, grads_list, w1_list, w2_list, bias_list = training_run(
        X_train, y_train, weights, bias, epochs, learning_rate=lr
    )
    axes[0].plot(loss_list, label=f'learning rate = {lr}')
    axes[1].plot(loss_test_list, label=f'learning rate = {lr}')

axes[0].set_title('train error')
axes[0].legend()

axes[1].set_title('test error')
axes[1].legend()



