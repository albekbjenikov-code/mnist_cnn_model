import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# Загружаем модель
model = load_model("/home/albek/ML/app/mnist_model.h5")

IMG_SIZE = 28
SCALE = 30  # во сколько раз увеличить для удобства рисования

class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Draw")

        # Холст для рисования
        self.canvas = tk.Canvas(
            root,
            width=IMG_SIZE * SCALE,
            height=IMG_SIZE * SCALE,
            bg="black"
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Изображение в памяти
        self.image = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Метка с предсказанием
        self.label = tk.Label(root, text="Нарисуй цифру", font=("Arial", 16))
        self.label.grid(row=0, column=1)

        # Вероятности
        self.prob_labels = []
        for i in range(10):
            lbl = tk.Label(root, text=f"{i}: 0.00")
            lbl.grid(row=i+1, column=1, sticky="w")
            self.prob_labels.append(lbl)

        # Кнопка очистки
        self.clear_btn = tk.Button(root, text="Очистить", command=self.clear)
        self.clear_btn.grid(row=11, column=1, pady=10)

        # События мыши
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x = event.x // SCALE
        y = event.y // SCALE

        r = 1
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        self.canvas.create_oval(
            (x-r)*SCALE, (y-r)*SCALE,
            (x+r)*SCALE, (y+r)*SCALE,
            fill="white", outline="white"
        )

        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, IMG_SIZE, IMG_SIZE), fill=0)
        self.label.config(text="Нарисуй цифру")
        for i in range(10):
            self.prob_labels[i].config(text=f"{i}: 0.00")

    def predict(self):
        img = np.array(self.image) / 255.0
        img = img.reshape(1, 28, 28, 1)

        preds = model.predict(img, verbose=0)[0]
        digit = np.argmax(preds)

        self.label.config(text=f"Предсказание: {digit}")

        for i in range(10):
            self.prob_labels[i].config(
                text=f"{i}: {preds[i]:.3f}"
            )

# Запуск
root = tk.Tk()
app = MNISTApp(root)
root.mainloop()
