import cv2

video = cv2.VideoCapture('C:\Python Project\Car_and_Pedestrian_Tracking\Videos\Pedestrians.mp4')

# Pre-Trained car and Pedestrian Classifiers
classifier_file = 'C:\Python Project\Car_and_Pedestrian_Tracking\Files\car_detector.xml'
classifier_pedestrian_file = 'C:\Python Project\Car_and_Pedestrian_Tracking\Files\pedestrian_detector.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(classifier_pedestrian_file)

# Run a loop until car stops
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 100, 100), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for(x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    cv2.imshow('Automatic Driving Car', frame)

    #Don't autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

video.release()