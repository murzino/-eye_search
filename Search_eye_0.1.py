### Может найти зрачки, присутствует много помех.

import cv2
import numpy as np

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

# Создание окна для отображения
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Применение оператора Кэнни для обнаружения краев


while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Применение размытия
    # gray = cv2.GaussianBlur(gray, (5, 5), 5)    


    # # Преобразование в Canny
    # gray = cv2.Canny(gray, threshold1=50, threshold2=50)

    # Использование функции HoughCircles для обнаружения кругов
    circles = cv2.HoughCircles(gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=100,
        # Как я понял определяет чувствительность к кругу
        param2=22,
        minRadius=13,
        maxRadius=32)

    # Если круги обнаружены, рисуем их на кадре
    if circles is not None:
        circles = np.uint16(circles)  # Преобразуем значения в целые
        # print(list(map(lambda x: sum(x)//len(x), zip(*circles[0, :]))))
        # i = list(map(lambda x: sum(x)//len(x), zip(*circles[0, :])))
        for i in circles[0,:]:

            # Рисуем закрашенную окружность
            cv2.circle(gray, (i[0], i[1]), i[2], (255, 255, 255), 1)  # Закрашенная окружность
            
            # Рисуем центр окружности
            cv2.circle(gray, (i[0], i[1]), 5, (255, 255, 255), -1)
            
            cv2.putText(gray, f'Count: {i}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

       

    # Отображаем кадр
    cv2.imshow('Video', gray)
    

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()