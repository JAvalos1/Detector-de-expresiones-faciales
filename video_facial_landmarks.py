# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

dist_smile0 = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
    	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		#cv2.putText(frame, "Face #{}".format(1), (x - 10, y - 10),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

	## Para detectar sonrisas
	dist_smile = ((shape[48][0]-shape[54][0])**2+(shape[48][0]-shape[54][1])**2)**0.5
	#diff_smile = dist_smile - dist_smile0
	if dist_smile>140:
		cv2.putText(frame, "Sonrisa", (x - 100, y - 50),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
	
	## Para detectar movimiento de las cejas
	dist_cejaD = ((shape[19][1]-shape[27][1])**2)**0.5
	dist_cejaI = ((shape[24][1]-shape[27][1])**2)**0.5
	print(dist_cejaD)
	print(dist_cejaI)

	if dist_cejaD>50 and dist_cejaI>50:
		cv2.putText(frame, "Ambas cejas levantadas", (x - 200, y - 250),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	elif dist_cejaD>50:
		cv2.putText(frame, "ceja derecha levantada", (x - 200, y - 250),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	elif dist_cejaI>50:
		cv2.putText(frame, "ceja izquierda levantada", (x - 200, y - 250),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	## Para detectar un beso
	dist_labios = ((shape[62][0]-shape[66][0])**2+(shape[62][1]-shape[66][1])**2)**0.5
	if dist_smile<98 and dist_labios<10:
		cv2.putText(frame, "Beso", (x - 60, y - 50),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

	## Para detectar ojos cerrados
	dist_ojoD = ((shape[37][1]-shape[41][1])**2)**0.5
	dist_ojoI = ((shape[44][1]-shape[46][1])**2)**0.5
	#print(dist_ojoD)
	#print(dist_ojoI)

	if dist_ojoD<12 and dist_ojoI<12:
		cv2.putText(frame, "Ambas ojos cerrados", (x - 150, y - 150),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	elif dist_ojoD<12:
		cv2.putText(frame, "ojo derecho cerrado", (x + 50, y - 150),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	elif dist_ojoI<12:
		cv2.putText(frame, "ojo izquierdo cerrado", (x-250, y - 150),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	#cv2.circle(frame, (shape[54][0], shape[54][1]), 5, (0, 255, 0), -1)
	#cv2.circle(frame, (shape[48][0], shape[48][1]), 5, (0, 255, 0), -1)  
	#print(shape[50][1])
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

## Para ejecutar: python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat