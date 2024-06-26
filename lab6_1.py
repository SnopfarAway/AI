import numpy as np
import matplotlib.pyplot as plt

class KohonenNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Инициализация весов случайными значениями
        self.weights = np.random.rand(output_size, input_size)

    def train(self, data, epochs):
        for epoch in range(epochs):
            np.random.shuffle(data)
            for point in data:
                # Находим ближайший нейрон кластера (победителя)
                winner_idx = np.argmin(np.linalg.norm(self.weights - point, axis=1))
                # Обновляем веса только для победившего нейрона
                learning_rate = (50 - epoch) / 100
                self.weights[winner_idx] += learning_rate * (point - self.weights[winner_idx])

    def classify(self, point):
        # Находим ближайший нейрон кластера (победителя)
        winner_idx = np.argmin(np.linalg.norm(self.weights - point, axis=1))
        return winner_idx

# Генерация случайных точек вокруг центров кластеров
center1 = np.array([2, 3])
center2 = np.array([7, 6])
data1 = center1 + np.random.randn(20, 2)
data2 = center2 + np.random.randn(20, 2)
data = np.vstack((data1, data2))

# Создание и обучение сети
network = KohonenNetwork(input_size=2, output_size=2)
network.train(data, epochs=50)

# Визуализация данных и результатов
plt.scatter(data[:, 0], data[:, 1], color='b', label='Data Points')
plt.scatter(network.weights[:, 0], network.weights[:, 1], color='r', marker='x', label='Cluster Centers')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kohonen Network Clustering')
plt.legend()

# Классификация новых точек
new_points = np.array([[3, 4], [6, 5], [4, 7], [8, 4]])
for point in new_points:
    cluster = network.classify(point)
    plt.scatter(point[0], point[1], color='g' if cluster == 0 else 'm', marker='o', label=f'Cluster {cluster}')

plt.show()
