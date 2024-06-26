import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import os
import cv2
import numpy as np

# Создаем главное окно приложения
root = tk.Tk()
root.title("Распознавание фигур")

# Переменные для подсчета количества каждого типа фигур
triangle_count = 0
circle_count = 0
square_count = 0

# Создаем функцию для отображения главного меню
def show_main_menu():
    # Очищаем окно
    for widget in root.winfo_children():
        widget.destroy()

    # Создаем кнопки главного меню
    button1 = tk.Button(root, text="Добавить изображение в обучающую выборку", command=show_training_menu)
    button1.pack()

    button2 = tk.Button(root, text="Распознать изображение", command=show_recognition_menu)
    button2.pack()

# Создаем функцию для отображения меню обучения
def show_training_menu():
    # Очищаем окно
    for widget in root.winfo_children():
        widget.destroy()

    # Создаем холст для рисования
    canvas = tk.Canvas(root, width=400, height=400, bg='white')
    canvas.pack()

    # Создаем объект Image и объект ImageDraw для рисования
    image = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(image)

    # Переменные для хранения предыдущих координат
    prev_x = None
    prev_y = None

    # Создаем функцию для рисования на холсте и на объекте Image
    def draw_on_canvas(event):
        nonlocal prev_x, prev_y
        x, y = event.x, event.y
        if prev_x is not None and prev_y is not None:
            canvas.create_line(prev_x, prev_y, x, y, fill="black", width=5)
            draw.line([(prev_x, prev_y), (x, y)], fill="black", width=5, joint='curve')
        prev_x = x
        prev_y = y

    # Привязываем функцию к событию нажатия мыши на холсте
    canvas.bind("<B1-Motion>", draw_on_canvas)

    # Создаем функцию для сохранения изображения
    def save_image(shape):
        global triangle_count, circle_count, square_count
        if not os.path.exists("training_images"):
            os.makedirs("training_images")

        if shape == "triangle":
            triangle_count += 1
            count = triangle_count
        elif shape == "circle":
            circle_count += 1
            count = circle_count
        elif shape == "square":
            square_count += 1
            count = square_count

        image.save(f"training_images/{shape}_{count}.png")
        messagebox.showinfo("Информация", "Изображение сохранено")
        show_main_menu()

    # Создаем кнопки для выбора фигуры
    button1 = tk.Button(root, text="Треугольник", command=lambda: save_image("triangle"))
    button1.pack()

    button2 = tk.Button(root, text="Круг", command=lambda: save_image("circle"))
    button2.pack()

    button3 = tk.Button(root, text="Квадрат", command=lambda: save_image("square"))
    button3.pack()

    # Создаем кнопку "Назад"
    button5 = tk.Button(root, text="Назад", command=show_main_menu)
    button5.pack()

# Создаем функцию для отображения меню распознавания
def show_recognition_menu():
    # Очищаем окно
    for widget in root.winfo_children():
        widget.destroy()

    # Создаем холст для рисования
    canvas = tk.Canvas(root, width=400, height=400, bg='white')
    canvas.pack()

    # Создаем объект Image и объект ImageDraw для рисования
    image = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(image)

    # Переменные для хранения предыдущих координат
    prev_x = None
    prev_y = None

    # Создаем функцию для рисования на холсте и на объекте Image
    def draw_on_canvas(event):
        nonlocal prev_x, prev_y
        x, y = event.x, event.y
        if prev_x is not None and prev_y is not None:
            canvas.create_line(prev_x, prev_y, x, y, fill="black", width=5)
            draw.line([(prev_x, prev_y), (x, y)], fill="black", width=5)
        prev_x = x
        prev_y = y

    # Привязываем функцию к событию нажатия мыши на холсте
    canvas.bind("<B1-Motion>", draw_on_canvas)

    # Создаем функцию для распознавания изображения
    def recognize_image():
        user_image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        best_match = None
        
        # Создаем словарь для хранения суммарных сходств для каждого типа изображения
        image_types = ["circle", "triangle", "square"]
        total_similarities = {image_type: 0 for image_type in image_types}

        # Проходим по всем тренировочным изображениям
        for filename in os.listdir("training_images"):
            # Загружаем тренировочное изображение
            test_image = cv2.imread(os.path.join("training_images", filename), cv2.IMREAD_GRAYSCALE)
            
            if test_image is not None:  # Проверяем, было ли изображение успешно загружено
                # Вычисляем схожесть между изображениями
                counter = np.sum(np.abs(user_image_gray - test_image))
                similarity = 10000000000000.0 / (1.0 + counter * counter)
                print(f"{filename}: {similarity}")
                # Добавляем сходство к соответствующему типу изображения
                for image_type in image_types:
                    if image_type in filename:
                        total_similarities[image_type] += similarity
                        break

        # Определяем тип изображения с наибольшим суммарным сходством
        best_match = max(total_similarities, key=total_similarities.get)
            
        # Выводим результаты
        if best_match:
            messagebox.showinfo("Результат", f"Наиболее схожая фигура: {best_match}")
        else:
            messagebox.showinfo("Результат", "Фигура не распознана")
            
        # Возвращаемся в главное меню после распознавания
        show_main_menu()

    # Создаем кнопку для распознавания изображения
    button = tk.Button(root, text="Распознать изображение", command=recognize_image)
    button.pack()

    # Создаем кнопку "Назад"
    button = tk.Button(root, text="Назад", command=show_main_menu)
    button.pack()

# Отображаем главное меню
show_main_menu()

# Запускаем главный цикл приложения
root.mainloop()
