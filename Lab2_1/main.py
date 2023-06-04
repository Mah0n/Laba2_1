import sys
import numpy as np
import cv2 as cv
import math

# Задаем диапазон зеленого цвета в HSV
hsv_min = np.array((35, 50, 50), np.uint8)
hsv_max = np.array((80, 255, 255), np.uint8)
color_blue = (255, 0, 0)
color_yellow = (0, 255, 255)

def detect_and_draw_rectangle(image):
    # Преобразуем изображение в цветовое пространство HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Применяем цветовой фильтр для выделения зеленого цвета
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    # Находим контуры объектов на бинарном изображении
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Вычисляем описывающий прямоугольник для каждого контура
        rect = cv.minAreaRect(cnt)
        # Получаем координаты вершин прямоугольника
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # Вычисляем площадь прямоугольника
        area = int(rect[1][0] * rect[1][1])
        # Вычисляем площадь контура
        area_a = cv.contourArea(cnt)
        # Вычисляем координаты центра прямоугольника
        center = (int(rect[0][0]), int(rect[0][1]))
        # Вычисляем векторы, задающие стороны прямоугольника
        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

        usedEdge = edge1
        if cv.norm(edge2) > cv.norm(edge1):
            usedEdge = edge2
        reference = (1, 0)

        if cv.norm(usedEdge) != 0:
            # Вычисляем угол между самой длинной стороной прямоугольника и горизонтом
            angle = 180.0 / math.pi * math.acos(
                (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv.norm(reference) * cv.norm(usedEdge))
            )

            if area > 1700:
                # Рисуем прямоугольник на исходном изображении
                cv.drawContours(image, [box], 0, (0, 255, 0), 2)  # Зеленый цвет
                # Выводим значение угла поворота на изображении
                cv.putText(
                    image,
                    "Angle: %d" % int(angle),
                    (center[0] + 20, center[1] - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    color_yellow,
                    2,
                )
                # Выводим координаты центра прямоугольника на изображении
                cv.putText(
                    image,
                    "Center: (%d, %d)" % (center[0], center[1]),
                    (center[0] + 20, center[1] + 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    color_yellow,
                    2,
                )

    return image

if __name__ == "__main__":
    # Захватываем видеопоток с камеры
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаруживаем и рисуем прямоугольники на каждом кадре
        processed_frame = detect_and_draw_rectangle(frame)
        # Выводим обработанный кадр на экран
        cv.imshow("Rectangle Detection", processed_frame)

        if cv.waitKey(1) == ord("q"):
            break

    # Освобождаем ресурсы
    cap.release()
    cv.destroyAllWindows()
