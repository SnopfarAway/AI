import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция для генерации точек с метками
def generate_points(n=10):
    points = []
    labels = []
    for i in range(1, 9):  
        for _ in range(n):
            x = np.random.uniform(1, 10)
            y = np.random.uniform(1, 10)
            z = np.random.uniform(1, 10)
            if i % 2 == 0:
                x = -x
            if (i//2) % 2 == 0:
                y = -y
            if i > 4:
                z = -z
            points.append([x, y, z])
            label = [0]*8
            label[i-1] = 1
            labels.append(label)
    return np.array(points), np.array(labels)

# Сигмоидальная функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоидальной функции для обратного распространения ошибки
def sigmoid_derivative(x):
    return x * (1 - x)

# Класс, реализующий перцептрон
class Perceptron:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        # 1 Инициализация весов случайными значениями
        self.weights = np.random.uniform(-1, 1, (input_dim, output_dim)) 
        self.bias = np.random.uniform(-1, 1, output_dim) #Смещение позволяет нейронной сети более гибко настраиваться на данные, включая смещение относительно нулевого входа.
        self.learning_rate = learning_rate

    # Метод предсказания
    def predict(self, inputs):
        # 2-3 Вычисление сигнала NET путем матричного перемножения входов и весов и применение сигмоидальной (пороговой) функции активации NET от каждого нейрона 
        return sigmoid(np.dot(inputs, self.weights) + self.bias)

    def train(self, training_inputs, labels, target_error=1):
        error = float('inf')  # Начальное значение ошибки
        epoch = 0  # Счетчик эпох
        while error > target_error:  # Повторять, пока ошибка больше, чем целевое значение ошибки
            error_sum = 0  # Обнуляем суммарную ошибку на каждой эпохе
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * np.dot(inputs.reshape(-1, 1), error.reshape(1, -1)) * sigmoid_derivative(prediction)
                self.bias += self.learning_rate * error * sigmoid_derivative(prediction)
                error_sum += np.abs(error).sum()  # Суммируем абсолютные значения ошибки
            epoch += 1
            error = error_sum / len(training_inputs)  # Средняя ошибка на эпохе
            print(f"Epoch {epoch}, Error: {error}")


# Генерация точек для обучения
points, labels = generate_points(10)

# Создание и тренировка персептрона
perceptron = Perceptron(input_dim=3, output_dim=8)
perceptron.train(points, labels)

# Создание окна для графиков
fig = plt.figure(figsize=(12, 6))

# Построение первого графика (исходное разбиение пространства)
ax1 = fig.add_subplot(121, projection='3d')

# Исходное разбиение пространства
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'purple']
for i in range(8):
    points_i = points[i*10:(i+1)*10]
    ax1.scatter(points_i[:,0], points_i[:,1], points_i[:,2], c=colors[i], label=f'Class {i+1}')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Первоначальное разбиение')
ax1.legend()

# Построение второго графика (классифицированные точки)
ax2 = fig.add_subplot(122, projection='3d')

# Классифицированные точки
predicted_labels = np.argmax(perceptron.predict(points), axis=1)
for i in range(8):
    points_i = points[predicted_labels == i]
    ax2.scatter(points_i[:,0], points_i[:,1], points_i[:,2], c=colors[i], label=f'Class {i+1}')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Классифицированные точки')
ax2.legend()

# Отображение обоих графиков
plt.show()
