#Перешел на встроеную библиотеку, тречится лучше

import cv2
import mediapipe as mp

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                   max_num_faces=1, 
                                   refine_landmarks=True, 
                                   min_detection_confidence=0.5)



eye_indices = [7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 56, 111, 112, 113, 114, 130, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 168, 173, 188, 189, 190, 193, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 243, 244, 245, 245, 246, 247, 249, 252, 253, 254, 255, 256, 257, 258, 259, 260, 263, 286, 339, 341, 359, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 412, 413, 414, 441, 442, 443, 444, 445, 446, 447, 448, 450, 451, 452, 453, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477]




# Размер области, которую мы хотим захватить вокруг лица
crop_width, crop_height = 500, 230


# Открытие подключения к камере
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование BGR в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка кадра для определения ключевых точек
    results = face_mesh.process(rgb_frame)

    # Внутри вашего цикла обработки, вместо рисования всех точек
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Рисуем определенные ключевые точки
            for index in eye_indices:


                # Вычисляем координаты центра лица (например, используем координаты носа)
                h, w, _ = frame.shape
                x_center = int(face_landmarks.landmark[168].x * w) # Используем индекс 1 для носа
                y_center = int(face_landmarks.landmark[168].y * h)

                # Вычисляем координаты левого глаза по 3м точкам 469 471
                x_eye_center_L = round(((face_landmarks.landmark[468].x * w) + (face_landmarks.landmark[469].x * w) + (face_landmarks.landmark[471].x * w)) / 3)
                y_eye_center_L = round(((face_landmarks.landmark[468].y * h) + (face_landmarks.landmark[469].y * h) + (face_landmarks.landmark[471].y * h)) / 3)

                # Вычисляем координаты правого глаза
                x_eye_center_R = round(((face_landmarks.landmark[473].x * w) + (face_landmarks.landmark[476].x * w) + (face_landmarks.landmark[474].x * w)) / 3)
                y_eye_center_R = round(((face_landmarks.landmark[473].y * h) + (face_landmarks.landmark[476].y * h) + (face_landmarks.landmark[474].y * h)) / 3)

                # 51 58 45 447 227 197 195
                # Вычисляем средние координаты по лицу которые не изменяются относительно глаз
                x_stability = round(((face_landmarks.landmark[51].x * w) + (face_landmarks.landmark[58].x * w) + (face_landmarks.landmark[45].x * w) + (face_landmarks.landmark[447].x * w) + (face_landmarks.landmark[227].x * w) + (face_landmarks.landmark[197].x * w) + (face_landmarks.landmark[195].x * w)) / 7)

                y_stability = round(((face_landmarks.landmark[51].y * h) + (face_landmarks.landmark[58].y * h) + (face_landmarks.landmark[45].y * h) + (face_landmarks.landmark[447].y * h) + (face_landmarks.landmark[227].y * h) + (face_landmarks.landmark[197].y * h) + (face_landmarks.landmark[195].y * h)) / 7)

                # Вычисляем координаты обрезанной области
                x1 = max(x_center - crop_width // 2, 0)
                y1 = max(y_center - crop_height // 2, 0)
                x2 = min(x_center + crop_width // 2, w)
                y2 = min(y_center + crop_height // 2, h)

                # Обрезаем изображение
                cropped_frame = frame[y1:y2, x1:x2]

                # Отображение обрезанного изображения
                cv2.imshow('Cropped Face Region', cropped_frame)

                # Координаты и отображение точек на лице
                landmark = face_landmarks.landmark[index]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Рисуем круги только для указанных точек

                # Показать координаты зрачка L
                cv2.putText(frame, f'left: {x_eye_center_L, y_eye_center_L}', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

                # Показать координаты зрачка R
                cv2.putText(frame, f'righ: {x_eye_center_R, y_eye_center_R}', (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

                # Показать координаты cтабильной точки
                cv2.putText(frame, f'stability: {x_stability, y_stability}', (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

                ### Координаты разницы
                x_see = (x_stability - x_eye_center_L)  * 35 - 4000
                y_see = (y_stability  - y_eye_center_L)

                # Показать Разницу
                cv2.putText(frame, f'raise: {x_see, y_see}', (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

                # рисуем круг куда смотрит глаз
                cv2.circle(frame, (x_see + 500, y_see  + 200), 30, (0, 0, 0), -1)  # Рисуем круги только для указанных точек

                # # Показать номер точек
                # cv2.putText(frame, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Опционально: рисуем соединения между ключевыми точками для визуализации
            # connections = mp_face_mesh.FACEMESH_TESSELATION  # Использование созданных соединений
            # mp.solutions.drawing_utils.draw_landmarks(
            #     frame, face_landmarks, connections)

    # Отображение результирующего кадра
    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()