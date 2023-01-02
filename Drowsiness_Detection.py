from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
	A = distance.euclidean(mouth[2], mouth[10])
	B = distance.euclidean(mouth[4], mouth[8])  
	C = distance.euclidean(mouth[0], mouth[6]) 
	mar = (A + B) / (2.0 * C)
	return mar
	
MOUTH_AR_THRESH = 0.85
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)
cap=cv2.VideoCapture(0)
eye_flag=0
mouth_flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		mouth = shape[mStart:mEnd]
		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)

		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			mouth_flag += 1
			print("Mouth Flag: " + str(mouth_flag))
			# print(mar)
			if mouth_flag >= frame_check:
				cv2.putText(frame, "****************YAWNING!****************", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************YAWNING!****************", (10,325),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
		else:
			mouth_flag = 0

		if ear < thresh:
			eye_flag += 1
			print ("Eye Flag: " + str(eye_flag))
			if eye_flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
		else:
			eye_flag = 0

		 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 
