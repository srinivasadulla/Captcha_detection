import cv2
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-m", "--model", required=True, help="path to the trained model file")

args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(args["model"])

print("[INFO] preprocessing input image...")
image = cv2.imread(args["image"])
cv2.imshow("Original image",image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
thresh = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours[0], key=cv2.contourArea, reverse=True)[:5]
output = cv2.merge([gray] * 3)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    img = gray[y - 5 : y + h + 5, x - 5 : x + w + 5]
    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    cv2.imshow("img", img)
    cv2.waitKey()
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img).argmax(axis=1)[0] + 1
    cv2.rectangle(output, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
    cv2.putText(output, str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imshow("Output", output)
cv2.waitKey()
