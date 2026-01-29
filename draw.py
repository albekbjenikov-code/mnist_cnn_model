import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

class MNISTDrawingApp:
    def __init__(self, model_path="/home/albek/ML/app/mnist_model.h5"):
        # Проверка существования модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
            
        print(f"Загрузка модели из: {model_path}")
        self.model = load_model(model_path)
        
        self.root = tk.Tk()
        self.root.title("MNIST Drawing App")
        
        # Увеличиваем окно
        self.root.geometry("900x900")
        
        # Инициализация атрибутов
        self.line_width = 30  # толщина линии по умолчанию
        self.canvas_size = 840  # 280 * 3 = 840 пикселей
        
        self.setup_ui()
        self.setup_bindings()
        
        # Инициализация изображения
        self.init_image()
        
    def init_image(self):
        """Инициализация изображения"""
        self.current_image = np.ones((self.canvas_size, self.canvas_size), dtype=np.uint8) * 255  # белый фон
        
    def setup_ui(self):
        """Настройка интерфейса с центровкой"""
        # Центрируем окно на экране
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Главный контейнер с отступами
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(main_container, text="Распознавание рукописных цифр MNIST", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Фрейм для canvas с центровкой
        canvas_frame = ttk.Frame(main_container)
        canvas_frame.pack(expand=True)
        
        # Canvas для рисования (в 3 раза больше - 840x840)
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg='white', cursor="cross", highlightthickness=2, 
                                highlightbackground="gray")
        self.canvas.pack()
        
        # Метка для результата (под canvas)
        self.result_label = ttk.Label(main_container, text="Нарисуйте цифру в белом поле выше", 
                                      font=("Arial", 14))
        self.result_label.pack(pady=20)
        
        # Фрейм для элементов управления
        control_frame = ttk.Frame(main_container)
        control_frame.pack(pady=20)
        
        # Кнопки (центрированные)
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Распознать", 
                  command=self.predict_drawing, width=15).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Очистить", 
                  command=self.clear_canvas, width=15).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Выход", 
                  command=self.root.quit, width=15).pack(side=tk.LEFT, padx=10)
        
        # Фрейм для настройки кисти
        brush_frame = ttk.LabelFrame(control_frame, text="Настройка кисти", padding=10)
        brush_frame.pack(pady=10, fill=tk.X)
        
        # Слайдер для толщины кисти
        brush_controls = ttk.Frame(brush_frame)
        brush_controls.pack()
        
        ttk.Label(brush_controls, text="Толщина:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.brush_slider = tk.Scale(brush_controls, from_=5, to=45, orient=tk.HORIZONTAL,
                                    length=300, command=self.update_brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=10)
        self.brush_slider.set(self.line_width)
        
        # Показатель текущей толщины
        self.brush_value_label = ttk.Label(brush_controls, text=f"{self.line_width}px", 
                                          width=6)
        self.brush_value_label.pack(side=tk.LEFT, padx=5)
        
        # Информационная панель
        info_frame = ttk.Frame(main_container)
        info_frame.pack(pady=20)
        
        # Информация о модели
        model_info = ttk.Label(info_frame, 
                              text="✓ Модель MNIST загружена и готова к работе",
                              font=("Arial", 10, "italic"))
        model_info.pack()
        
        # Инструкция
        instruction = ttk.Label(info_frame,
                               text="Инструкция: Нарисуйте цифру от 0 до 9 и нажмите 'Распознать'",
                               font=("Arial", 9))
        instruction.pack(pady=5)
        
    def setup_bindings(self):
        """Настройка обработчиков событий"""
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drawing)
        
    def update_brush_size(self, value):
        """Обновление толщины кисти"""
        self.line_width = int(value)
        self.brush_value_label.config(text=f"{self.line_width}px")
        
    def draw(self, event):
        """Рисование на холсте"""
        x, y = event.x, event.y
        
        # Рисуем на холсте
        radius = self.line_width // 2
        self.canvas.create_oval(x - radius, y - radius, 
                               x + radius, y + radius,
                               fill='black', outline='black')
        
        # Обновляем массив изображения
        if self.current_image is None:
            self.init_image()
            
        # Рисуем на массиве (белый фон, черная цифра)
        cv2.circle(self.current_image, (x, y), self.line_width, 0, -1)  # 0 = черный цвет
        
    def reset_drawing(self, event):
        """Сброс рисования"""
        pass
        
    def clear_canvas(self):
        """Очистка холста"""
        self.canvas.delete("all")
        # Очищаем холст белым цветом
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, 
                                    fill='white', outline='white')
        self.init_image()
        self.result_label.config(text="Нарисуйте цифру в белом поле выше")
        
    def preprocess_image(self, img):
        """Предобработка изображения для модели"""
        # 1. Изменить размер с 840x840 до 28x28
        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 2. Инвертировать цвета (черная цифра на белом фоне → белая на черном)
        img_inverted = 255 - img_resized
        
        # 3. Нормализовать
        img_normalized = img_inverted / 255.0
        
        # 4. Добавить размерности (batch, height, width, channels)
        img_processed = img_normalized.reshape(1, 28, 28, 1)
        
        # Для отладки: сохранить обработанное изображение
        # cv2.imwrite("processed_debug.png", img_inverted)
        
        return img_processed
        
    def predict_drawing(self):
        """Распознавание нарисованной цифры"""
        if self.current_image is None or np.all(self.current_image == 255):
            messagebox.showwarning("Внимание", "Сначала нарисуйте цифру!")
            return
            
        try:
            # Сохраняем текущий рисунок для отладки
            # cv2.imwrite("current_drawing.png", self.current_image)
            
            # Предобработка изображения
            img_processed = self.preprocess_image(self.current_image)
            
            # Предсказание
            predictions = self.model.predict(img_processed, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Форматируем текст результата
            result_text = f"Распознанная цифра: {predicted_class}\n"
            result_text += f"Уверенность: {confidence:.1%}"
            
            # Цвет текста в зависимости от уверенности
            if confidence > 0.9:
                text_color = "green"
            elif confidence > 0.7:
                text_color = "orange"
            else:
                text_color = "red"
                
            # Обновление интерфейса
            self.result_label.config(
                text=result_text,
                font=("Arial", 14, "bold"),
                foreground=text_color
            )
            
            # Отладочная информация
            print(f"\n{'='*50}")
            print(f"Результаты предсказания:")
            print(f"Цифра: {predicted_class}")
            print(f"Уверенность: {confidence:.2%}")
            print("Все вероятности:")
            for i, prob in enumerate(predictions[0]):
                stars = "★" * int(prob * 20)
                print(f"  {i}: {prob:.4f} {stars}")
            print('='*50)
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось распознать цифру:\n{str(e)}")
            print(f"Ошибка: {e}")
            
    def save_drawing(self):
        """Сохранение нарисованного изображения"""
        if self.current_image is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_{timestamp}.png"
            cv2.imwrite(filename, self.current_image)
            messagebox.showinfo("Сохранено", f"Рисунок сохранен как:\n{filename}")
            print(f"Изображение сохранено как {filename}")
            
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

def main():
    import time
    
    # Укажите абсолютный путь к вашей модели
    model_path = "/home/albek/ML/app/mnist_model.h5"
    
    # Проверка существования файла модели
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл модели не найден по пути: {model_path}")
        print("Проверьте:")
        print(f"1. Существует ли файл: ls {model_path}")
        print("2. Установите модель если её нет:")
        print("   wget https://example.com/mnist_model.h5 -O /home/albek/ML/app/mnist_model.h5")
        return
        
    try:
        # Создание и запуск приложения
        print("Запуск приложения распознавания цифр MNIST...")
        print(f"Размер холста: 840x840 пикселей (в 3 раза больше стандартного)")
        
        app = MNISTDrawingApp(model_path)
        
        print("\nПриложение запущено успешно!")
        print("="*50)
        print("ИНСТРУКЦИЯ:")
        print("1. Нарисуйте цифру в большом белом поле")
        print("2. Используйте слайдер для изменения толщины кисти")
        print("3. Нажмите 'Распознать' для предсказания")
        print("4. Нажмите 'Очистить' для очистки холста")
        print("="*50)
        
        app.run()
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("\nУстановите необходимые библиотеки:")
        print("pip install opencv-python tensorflow numpy")
        
    except Exception as e:
        print(f"Ошибка при запуске приложения: {e}")

if __name__ == "__main__":
    main()