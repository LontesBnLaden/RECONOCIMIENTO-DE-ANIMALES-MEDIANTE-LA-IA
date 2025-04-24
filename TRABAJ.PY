import cv2

# Cargar el clasificador Haar para gatos (frontal)
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

if cat_cascade.empty():
    print("No se pudo cargar el clasificador de gatos.")
    exit()

# Captura desde cámara
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # ancho
video_capture.set(4, 480)  # alto

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detección de gatos
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    
    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, 'Gato detectado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Detección de bordes
    edges = cv2.Canny(gray, 100, 200)
    cv2.imshow('Detección de Bordes', edges)

    # Mostrar la imagen con la detección
    cv2.imshow('Detección de Gatos', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

