import numpy as np
import PIL
from PIL import Image
import os
import shutil

def read_image(filename):
    try:
        image = np.array(Image.open(filename))
        return image.reshape(-1)  # преобразовать изображения в одномерный массив
    except (PIL.UnidentifiedImageError, OSError): # если есть мусор
        print(f"Ignore file: {filename}")
        return None

def initialize_weights(input_size, output_size):
    return np.random.rand(output_size, input_size) # задать веса случайными значениями

# функция для поиска наилучшего совпадения
def find_bmu(input_vector, weights):
    distances = np.linalg.norm(weights - input_vector, axis=1) #вычислить Евклидово расстояние между входным вектором и каждым вектором весов в сети. 
    return np.argmin(distances) #вернуть индекс наименьшего элемента в массиве расстояний

def update_weights(input_vector, weights, bmu_index, learning_rate, radius):
    for i, weight in enumerate(weights):
        distance_to_bmu = np.abs(i - bmu_index) # расчитать расстояние от текущего нейрона до нейрона победителя
        if distance_to_bmu <= radius:
            influence = np.exp(-(distance_to_bmu ** 2) / (2 * radius ** 2)) # используется для того, чтобы обновление весов нейрона было не равномерным, а зависело от расстояния до BMU
            weights[i] += learning_rate * influence * (input_vector - weight)

# функция для обучения SOM
def train_som(input_data, output_size, epochs, learning_rate_initial, radius_initial):
    input_size = input_data.shape[1]
    weights = initialize_weights(input_size, output_size)
    for epoch in range(epochs):
        # постепенное уменьшение learning rate и радиуса
        learning_rate = learning_rate_initial * (1 - epoch / epochs)
        radius = radius_initial * (1 - epoch / epochs)
        for input_vector in input_data:
            bmu_index = find_bmu(input_vector, weights)
            update_weights(input_vector, weights, bmu_index, learning_rate, radius)
    return weights

image_folder = "training_images"
image_files = os.listdir(image_folder)
images = [read_image(os.path.join(image_folder, img)) for img in image_files]
images = [img for img in images if img is not None] # убрать из выборки пустые значения

num_classes = 3
epochs = 100
learning_rate_initial = 0.5  # начальный learning rate
radius_initial = num_classes / 2  # начальный радиус
input_data = np.array(images) # преобразовать изображения в массив NumPy

# обучить нейросеть
weights = train_som(input_data, num_classes, epochs, learning_rate_initial, radius_initial)

# кластеризовать изображения
clustered_images = [[] for _ in range(num_classes)]
for image, filename in zip(input_data, image_files):
    bmu_index = find_bmu(image, weights)
    clustered_images[bmu_index].append(filename)

# распределить изображения по папкам
for i, image_list in enumerate(clustered_images):
    folder_name = f"class_{i + 1}"
    os.makedirs(folder_name, exist_ok=True)
    for img_file in image_list:
        shutil.copy(os.path.join(image_folder, img_file), os.path.join(folder_name, img_file))
