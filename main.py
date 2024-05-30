import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from memory_profiler import memory_usage
from tabulate import tabulate

np.random.seed(100)


def generate_data(n_samples, n_features, noise):
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = 0.5 * (X ** 2).dot(true_coef) + X.dot(true_coef) + noise * np.random.randn(n_samples)
    if n_features == 1:
        X = np.squeeze(X)
    return X, y, true_coef


def generate_data_2():
    true_slope = 2
    true_intercept = 1
    X = np.random.rand(100, 1)
    noise = np.random.randn(100, 1) * 0.1
    y = true_slope * X + true_intercept + noise
    return X, y, true_slope, true_intercept


def compute_gradient(x, y, theta):
    prediction = x.dot(theta)
    error = prediction - y
    gradient = 2 * x.T.dot(error) / len(y)
    return gradient


def polynomial_regression(x, y, theta, learning_rate, iterations, regularization, coeff, batch_size, power=100):
    m = len(y)
    X = np.array([x ** i for i in range(power)]).T
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))
    operation_count = 0
    for it in range(iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index + batch_size]
        yi = y[random_index:random_index + batch_size]

        gradient = compute_gradient(xi, yi, theta)

        gradient += coeff * regularization['gradient'](theta)

        operation_count += batch_size

        theta = theta - learning_rate * gradient

        cost = (1.0 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2) + coeff * regularization['reg'](theta)
        cost_history[it] = cost
        theta_history[it, :] = theta.T

    return theta, cost_history, theta_history


def loss_function(prediction, actual):
    tf.reduce_mean(tf.squared_difference(prediction, actual))


def train_model(optimizer, X, y, epochs=100):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(X, y, epochs=epochs, verbose=0)
    prediction = model.predict(X)
    return model, history, prediction


def sgd_with_nesterov(params):
    return tf.keras.optimizers.SGD(learning_rate=params['learning_rate'],
                                   nesterov=params['nesterov'])


def sgd_with_momentum(params):
    return tf.keras.optimizers.SGD(learning_rate=params['learning_rate'],
                                   momentum=params['momentum'])


def sgd(params):
    return tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])


def adagrad(params):
    return tf.keras.optimizers.Adagrad(learning_rate=params['learning_rate'])


def rmsprop(params):
    return tf.keras.optimizers.RMSprop(learning_rate=params['learning_rate'])


def adam(params):
    return tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])


colors = {
    'SGD': 'blue',
    'SGD with Nesterov': 'green',
    'SGD with Momentum': 'red',
    'AdaGrad': 'cyan',
    'RMSprop': 'magenta',
    'Adam': 'yellow',
}

optimizers = {
    'SGD': sgd,
    'SGD with Nesterov': sgd_with_nesterov,
    'SGD with Momentum': sgd_with_momentum,
    'AdaGrad': adagrad,
    'RMSprop': rmsprop,
    'Adam': adam,
}

hyperparameters = {
    'SGD': {'learning_rate': 0.01, 'decay': 1e-6},
    'SGD with Nesterov': {'learning_rate': 0.01, 'nesterov': True},
    'SGD with Momentum': {'learning_rate': 0.01, "momentum": 0.9},
    'AdaGrad': {'learning_rate': 0.5},
    'RMSprop': {'learning_rate': 0.01},
    'Adam': {'learning_rate': 0.01},
}

histories = {}
predictions = {}

stats = {}


def print_table(stats, histories):
    table = []
    headers = ['Optimizer', 'Time (s)', 'Memory (MB)', 'Final Loss']

    for name, stat in stats.items():
        final_loss = histories[name][-1] if len(histories[name]) != 0 else None
        row = [name, stat[0], stat[1], final_loss]
        table.append(row)

    print(tabulate(table, headers, tablefmt='pipe', floatfmt=".2f"))


def draw_task_3():
    X, y, slope, intercept = generate_data_2()
    for name, optimizer in optimizers.items():
        start_time = time.time()
        start_mem = memory_usage()[0]
        _, history, prediction = train_model(optimizer(hyperparameters[name]), X, y, epochs=100)
        end_time = time.time()
        end_mem = memory_usage()[0]

        time_taken = end_time - start_time
        mem_used = end_mem - start_mem
        histories[name] = history.history['loss']
        predictions[name] = prediction
        stats[name] = [time_taken, mem_used, ]

    for name, prediction in predictions.items():
        plt.scatter(X, y)
        plt.plot(X, prediction, label=name, color=colors[name])
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f"Prediction for {name}")
        plt.legend()
        plt.show()

    plt.figure(figsize=(12, 8))
    for name, loss in histories.items():
        plt.plot(loss, label=name, color=colors[name])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch for Different Optimizers')
    plt.legend()
    plt.show()


best_l1_ratio = 0.9


def draw(X, y, theta, m_power, name, lam, threshold=10):
    X_new = np.array([X ** i for i in range(m_power)]).T
    y_pred = X_new.dot(theta)

    mask = (y_pred < threshold) & (y_pred > -threshold)

    y_pred_filtered = y_pred[mask]
    X_filtered = X[mask]

    plt.scatter(X_filtered, y_pred_filtered, color='red',
                label=f'Polynomial regression degree = {m_power - 1}, with {name} regularization, coeff = {lam}')
    plt.scatter(X, y)
    plt.legend()
    plt.show()


m_power = 5

regularization = {
    'L1': {'gradient': lambda x: np.sign(x), 'reg': lambda x: np.sum(np.abs(x)),
           'lambda': [10, 30, 35, 50, 70, 100, 130], 'iter': 450},
    'L2': {'gradient': lambda x: 2 * x, 'reg': lambda x: np.dot(x, x),
           'lambda': [10, 20, 40], 'iter': 10000},
    'Elastic': {
        'gradient': lambda x: (1 - best_l1_ratio) * np.sign(x) + best_l1_ratio * 2 * x,
        'reg': lambda x: (1 - best_l1_ratio) * np.sum(np.abs(x)) + best_l1_ratio * np.dot(x, x),
        'lambda': [25, 30, 40, 45, 51, 70],
        'iter': 2000
    },
}

regularization_stats = {}


def draw_dop_task_1():
    for name, reg in regularization.items():
        for lam in reg['lambda']:
            start_time = time.time()
            start_mem = memory_usage()[0]
            theta, cost_history, _ = polynomial_regression(X, y,
                                                           np.random.randn(m_power), 0.01,
                                                           reg['iter'], reg,
                                                           lam, 30, m_power)
            end_time = time.time()
            end_mem = memory_usage()[0]
            time_taken = end_time - start_time
            mem_used = end_mem - start_mem
            regularization_stats[f'{name} with lambda={lam}'] = [time_taken, mem_used]
            histories[f'{name} with lambda={lam}'] = cost_history
            draw(X, y, theta, m_power, name, lam)
    print_table(regularization_stats, histories)


X, y, _ = generate_data(100, 1, 0.1)

draw_dop_task_1()
