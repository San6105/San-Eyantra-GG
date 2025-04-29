import cv2
import numpy as np
from tensorflow import keras

detected_list=[]
keyy=["A","B","C","D","E"]
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
text="nothing"
def classify_event(image):
    model_path = r'G:\Documents\E-yantra\2023(Geoguide)\Task 2\Task 2B\Cnn.h5'  # Update with your model path
    model = keras.models.load_model(model_path)
    img_array = cv2.resize(image, (224, 224))  # Resize the image
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0) 
    predictions = model.predict(img_array)
    event_classes = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]  # Update with your event class labels
    predicted_class_index = np.argmax(predictions)
    predicted_event = event_classes[predicted_class_index]
    return predicted_event
def classification(event_list):
    global text
    detected_event = classify_event(img)
    if detected_event == combat:
        text="combat"
        detected_list.append(5)
    if detected_event == rehab:
        text="human_aid_rehabilitation"
        detected_list.append(3)
    if detected_event == military_vehicles:
        text="military_vehicles"
        detected_list.append(4)
    if detected_event == fire:
        text="fire"
        detected_list.append(1)
    if detected_event == destroyed_building:
        text="destroyed_building"
        detected_list.append(2)
def task_4a_return():
    identified_labels = {}  
    detected_list.sort()
    for i in range(5):
        if detected_list[i]==1:
            identified_labels[keyy[i]]="fire"
        elif  detected_list[i]==2:
            identified_labels[keyy[i]]="destroyed_buildings"
        elif detected_list[i]==3:
            identified_labels[keyy[i]]="human_aid_rehabilitation"
        elif detected_list[i]==4:
            identified_labels[keyy[i]]="military_vehicles"
        elif detected_list[i]==5:
            identified_labels[keyy[i]]="combat"
    return identified_labels
if __name__ == "__main__":
    image = cv2.imread(r'G:\Documents\E-yantra\2023(Geoguide)\Task 2\Task 2B\WhatsApp Image 2023-12-28 at 16.37.23_2031fed5.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > 500 and area < 200000: 
                    if any(cv2.norm(approx - s) < 250 for s in squares):
                        continue 
                    squares.append(approx)
    squares = sorted(squares, key=cv2.contourArea, reverse=True)[:5] 
    for i,square in enumerate(squares):
        x,y,w,h= cv2.boundingRect(square)
        img=image[y:y+h,x:x+w]
        classification(img)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('s',image)
    cv2.waitKey(0)
    identified_labels = task_4a_return()
    print(identified_labels)
    




