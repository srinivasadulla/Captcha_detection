import numpy
import argparse
import cv2
import os
from imutils import paths


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to image directory")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())


imgPaths = list(paths.list_images(args["directory"]))
counts = {}

for (i, imagepath) in enumerate(imgPaths):

    print("[INFO] processing image {}/{}...".format(i+1, len(imgPaths)))

    try:
        img = cv2.imread(imagepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours[0]:
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(gray, (x-5,y-5), (x+w+5, y+h+5), (255,255,255), 1)
            cropImage = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            cropImage = cv2.resize(cropImage, (28, 28), cv2.INTER_AREA)
            cv2.imshow("img", cropImage)
            key = cv2.waitKey(0)
            if key == "~":
                print("[INFO] number ignored...")
                continue

            key = chr(key).upper()
            dir = os.path.sep.join([args["output"], key])

            if not os.path.exists(dir):
                os.makedirs(dir)

            count = counts.get(key, 1)

            path = os.path.sep.join([dir, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(path, cropImage)
            counts[key] = count + 1

    except:
        print("error")
