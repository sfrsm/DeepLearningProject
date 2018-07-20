import ffmpy
import subprocess
import json
import cv2
from Image2DCT.Image2DCT import Image2DCT

video = "1_1_Y.flv"

fp = ffmpy.FFprobe(inputs={video: ['-show_entries', 'frame=pict_type,coded_picture_number', '-of', 'json']})

stdout, stderror = fp.run(stdout=subprocess.PIPE)

d = json.loads(stdout)

frameList = list()

for i in d['frames']:
    if len(i) != 0 and i['pict_type'] == "I":
        frameList.append(i['coded_picture_number'])
        continue

cap = cv2.VideoCapture(video)

for frameNumber in frameList:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber+1)
        ret, frame = cap.read()

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fileName = 'dctImage' + str(frameNumber) + '.jpg'

        dct = Image2DCT(gray)

        cv2.imwrite(fileName, dct)

cap.release()