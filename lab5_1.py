import numpy as np
import matplotlib.pyplot as plt

# Генерация случайных точек
num_points = 100
np.random.seed(0)
points = np.random.rand(num_points, 2) * 10  # 100 точек в диапазоне от 0 до 10

# Определяем область
xmin, xmax, ymin, ymax = 6, 10, 6, 10
area_points = (xmin <= points[:, 0]) & (points[:, 0] <= xmax) & (ymin <= points[:, 1]) & (points[:, 1] <= ymax)

# Метки классов: 1 если точка внутри области, 0 если снаружи
labels = area_points.astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_perceptron(X, Y, alpha, epochs):
    W = np.random.rand(2)
    for _ in range(epochs):
        for i in range(X.shape[0]):
            y_pred = sigmoid(np.dot(W, X[i])) # Вычисление значения пороговой функции активации
            error = Y[i] - y_pred # Вычисление значения ошибки
            W += alpha * error * X[i]  # Обновление весов
    return W


# Нормализуем точки и обучаем перцептрон
norm_points = points / 10.0
weights = train_perceptron(norm_points, labels, alpha=0.00001, epochs=100)
print(weights)
# Классификация новой точки
new_point = np.array([5,6])
norm_new_point = new_point / 10
predicted_label = sigmoid(np.dot(weights, norm_new_point))
print(predicted_label)
# Определение класса для визуализации
predicted_class = 'inside' if predicted_label > 0.59 else 'outside'

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='cool', label='Initial Points')
plt.scatter(*new_point, color='red', label=f'New Point ({predicted_class})')
plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-', label='Defined Area')
plt.title('Perceptron Classification')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.show()