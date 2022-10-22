import requests # get access to html pages
import os
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to images directory")
ap.add_argument("-n", "--num_images", default=500, help="number of images to be downloaded")
args = vars(ap.parse_args())


web = "https://www.e-zpassny.com/vector/jcaptcha.do"

for i in range(0, args["num_images"]):

    try:
        r = requests.get(web, timeout=60)
        print(r.content)
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(i).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()
        print("[INFO] downloaded:{}".format(p))
    except:
        print("[INFO] error downloading image...")
