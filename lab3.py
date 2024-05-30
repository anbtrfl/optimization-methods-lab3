import time

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
from tabulate import tabulate

np.random.seed(100)


def generate_data(n_samples, n_features, noise):
    X = np.random.randn(n_samples, n_features)
    true_intercept = 2
    true_coef = 2
    if n_features == 1:
        X = np.squeeze(X)
    y = X * true_coef + true_intercept + noise * np.random.randn(n_samples)
    return X, y, true_coef


def compute_gradient(x, y, theta):
    prediction = x.dot(theta)
    error = prediction - y
    gradient = 2 * x.T.dot(error) / len(y)
    return gradient


def stochastic_gradient_descent(X, y, theta, learning_rate, iterations, batch_size):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))

    for it in range(iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index + batch_size]
        yi = y[random_index:random_index + batch_size]

        gradient = compute_gradient(xi, yi, theta)

        theta = theta - learning_rate * gradient

        cost = (1.0 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)
        cost_history[it] = cost
        theta_history[it, :] = theta.T

    return theta, cost_history, theta_history


def stochastic_gradient_descent_with_decay(X, y, theta, learning_rate, decay_rate, iterations, batch_size):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))

    for it in range(iterations):
        indexes = np.random.choice(m, batch_size, replace=False)
        xi = X[indexes]
        yi = y[indexes]

        gradient = compute_gradient(xi, yi, theta)

        learning_rate_decay = learning_rate * np.exp(-decay_rate * it)
        theta = theta - learning_rate_decay * gradient

        cost = (1.0 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)
        cost_history[it] = cost
        theta_history[it, :] = theta.T

    return theta, cost_history, theta_history


def draw_different_batch():
    n_samples = 3000
    X, y, true_coef = generate_data(n_samples, 1, 0.1)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.random.randn(2)

    times = []
    memories = []
    final_losses = []
    batch_sizes = []
    for batch_size in range(1, n_samples, 50):
        start_time = time.time()
        start_mem = memory_usage()[0]
        theta, cost_history, theta_history = stochastic_gradient_descent(X_b, y, theta, learning_rate=0.01,
                                                                         iterations=200, batch_size=batch_size)
        end_time = time.time()
        end_mem = memory_usage()[0]
        times.append(end_time - start_time)
        memories.append(end_mem - start_mem)
        final_losses.append(cost_history[-1])
        batch_sizes.append(batch_size)

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes[1:], final_losses[1:], 'b.-')
    plt.title('Correlation between Batch Size and Loss')
    plt.xlabel('Batch Size')
    plt.ylabel('Final Loss')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, times, 'r.-')
    plt.title('Correlation between Batch Size and Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (s)')
    plt.show()

    table = list(zip(batch_sizes, times, memories, final_losses))
    print(tabulate(table, headers=['Batch Size', 'Time (s)', 'Memory (MB)', 'Final Loss'], tablefmt='pipe',
                   floatfmt=".5f"))


def draw_train_common(theta, cost_history, X, y, X_b, batch_size, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, X_b.dot(theta), color='red', label='Approximating Line' + title)
    plt.title(f'Approximating Line {title}, batch size = {batch_size}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, 'b.')
    plt.title(f'Convergence of Stochastic Gradient Descent {title}, batch size = {batch_size}')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def draw_train():
    n_samples = 3000
    iterations = 1000
    batch_size = 1
    X, y, true_coef = generate_data(n_samples, 1, 1)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.random.randn(2)
    theta, cost_history, theta_history = stochastic_gradient_descent(X_b, y, theta, learning_rate=0.01,
                                                                     iterations=iterations, batch_size=batch_size)
    draw_train_common(theta, cost_history, X, y, X_b, batch_size, '')


def draw_train_with_decay():
    n_samples = 3000
    iterations = 1000
    batch_size = 2000
    decay_rate = 0.05
    X, y, true_coef = generate_data(n_samples, 1, 0.1)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.random.randn(2)
    theta, cost_history, theta_history = stochastic_gradient_descent_with_decay(X_b, y, theta, learning_rate=0.1,
                                                                                decay_rate=decay_rate,
                                                                                iterations=iterations,
                                                                                batch_size=batch_size)
    draw_train_common(theta, cost_history, X, y, X_b, batch_size, ' with Decay')

    learning_rates = [0.1 * np.exp(-decay_rate * it) for it in range(iterations)]
    table = list(zip(range(iterations), learning_rates, cost_history))
    print(tabulate(table, headers=['Iteration', 'Learning Rate', 'Cost'], tablefmt='pipe', floatfmt=".5f"))


def draw_train_and_decay_comparison():
    n_samples = 3000
    iterations = 100
    batch_size = 100
    decay_rate = 5
    X, y, true_coef = generate_data(n_samples, 1, 0.1)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.random.randn(2)

    theta_sgd, cost_history_sgd, theta_history_sgd = stochastic_gradient_descent(X_b, y, theta, learning_rate=0.01,
                                                                                 iterations=iterations,
                                                                                 batch_size=batch_size)

    theta_decay, cost_history_decay, theta_history_decay = stochastic_gradient_descent_with_decay(X_b, y, theta,
                                                                                                  learning_rate=0.01,
                                                                                                  decay_rate=decay_rate,
                                                                                                  iterations=iterations,
                                                                                                  batch_size=batch_size)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history_sgd)), cost_history_sgd, 'b.-', label='SGD')
    plt.plot(range(len(cost_history_decay)), cost_history_decay, 'r.-', label='SGD with Decay')
    plt.title('Comparison of SGD and SGD with Decay')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()


def draw_different_decay_rate():
    n_samples = 3000
    X, y, true_coef = generate_data(n_samples, 1, 0.1)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.random.randn(2)

    times = []
    memories = []
    final_losses = []
    decay_rates = []
    for decay_rate in np.arange(0.1, 3, 0.1):
        start_time = time.time()
        start_mem = memory_usage()[0]
        theta, cost_history, theta_history = stochastic_gradient_descent_with_decay(X_b, y, theta, learning_rate=0.01,
                                                                                    decay_rate=decay_rate,
                                                                                    iterations=100, batch_size=100)
        end_time = time.time()
        end_mem = memory_usage()[0]
        times.append(end_time - start_time)
        memories.append(end_mem - start_mem)
        final_losses.append(cost_history[-1])
        decay_rates.append(decay_rate)

    plt.figure(figsize=(10, 6))
    plt.scatter(decay_rates, times, alpha=0.5)
    plt.title('Correlation between Decay Rate and Time')
    plt.xlabel('Decay Rate')
    plt.ylabel('Time (s)')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(decay_rates, final_losses, alpha=0.5)
    plt.title('Correlation between Decay Rate and Final Loss')
    plt.xlabel('Decay Rate')
    plt.ylabel('Final Loss')
    plt.show()

    table = list(zip(decay_rates, times, memories, final_losses))
    print(tabulate(table, headers=['Decay Rate', 'Time (s)', 'Memory (MB)', 'Final Loss'], tablefmt='pipe',
                   floatfmt=".8f"))


draw_different_batch()
