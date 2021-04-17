import os
import argparse
import cv2
from flask import Flask, render_template, Response,request,jsonify
from dotenv import load_dotenv
import urllib.request
import numpy as np
import json
import time

import imcut
from mitsuba.mycam import MyCamera

cutsize=(200,150)
allsize=(640,480)

#url="http://localhost:5000"
#camsrv="http://192.168.11.242:5000/feed"

bgtimeout = 180
bgtime = 0
bg = None

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


@app.route("/image")
def image():
    global bgtime
    global bg

    frame=cam.getframe()

    if frame is None : return Response(status=404)

    if time.time() - bgtime > bgtimeout : bg=None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if bg is None :
        bgtime = time.time()
        bg = gray

    frame_diff = cv2.absdiff(bg,gray)

    _,frame_diff = cv2.threshold(frame_diff,50,255, cv2.THRESH_BINARY)
    frame_diff = cv2.medianBlur(frame_diff, 5)

    diff_point=cv2.countNonZero(frame_diff)

    diff_thr=2000

    if diff_thr < diff_point:

        contrs,hierarchy = cv2.findContours(frame_diff,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for pt in contrs:
            mu = cv2.moments(pt)

            if(mu["m00"]>2000):
                cx=int(mu["m10"]/mu["m00"])
                cy=int(mu["m01"]/mu["m00"])

                cutwx,cutwy = imcut.adjust_size(mu["m00"],0.8,cutsize)

                xx = imcut.cut_over(cx,cutwx,allsize[0])
                yy = imcut.cut_over(cy,cutwy,allsize[1])

                expimg=frame[yy[0]:yy[1],xx[0]:xx[1]]
                #expimg = cv2.resize(expimg,cutsize)

                color=(0, 255, 0)

                if predict(expimg)>0.7 : color=(255, 0, 0)

                cv2.rectangle(frame,(xx[0],yy[0]),(xx[1],yy[1]),color, 1)

    _,jpeg = cv2.imencode('.jpg', frame)

    return Response(jpeg.tobytes(),mimetype="image/jpeg")


def predict(frame):

    #frame = cv2.resize(frame,(64,48))
    _,jpeg= cv2.imencode(".jpg", frame)

    req = urllib.request.Request(
        url,
        jpeg.tobytes(),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )

    response = urllib.request.urlopen(req)
    json_str = response.read()
    response.close()

    j = json.loads(json_str)

    return j["prob"]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-p","--port",type=int,default=5000)

    args = parser.parse_args()

    myport=int(args.port)

    load_dotenv()

    cam=MyCamera(os.getenv("CAMERA_SRV"))
    url=os.getenv("PREDICT_URL")

    app.run(host='0.0.0.0', debug=False,threaded=True,port=myport)
