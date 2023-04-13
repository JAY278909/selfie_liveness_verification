import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import cv2
import os
from tensorflow.keras import backend as k
import sys
from io import BytesIO

class Liveness_Verification():


	def verifyLiveness(self, img_bytes, request_parameter):

		response_liveness = {}

		try:
			response_liveness = {}
			nparr = np.fromstring(img_bytes, np.uint8)
			img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			net = cv2.dnn.readNet(
				os.getcwd() + '/Ml_Models/darknet_liveness_model/yolov3_liveness_custom_last.weights',
				os.getcwd() + '/Ml_Models/darknet_liveness_model/yolov3_liveness_custom.cfg')
			classes = []
			with open(os.getcwd() + "/Ml_Models/darknet_liveness_model/classes.names", "r") as f:
				classes = f.read().splitlines()
			boxes = []
			confidences = []
			class_ids = []
			layers_names = net.getLayerNames()
			output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
			image = cv2.resize(img_np, None, fx=0.4, fy=0.4)
			height, width, channel = image.shape
			blob = cv2.dnn.blobFromImage(image, 0.0039, (416, 416), (0, 0, 0), True, crop=False)
			net.setInput(blob)
			outs = net.forward(output_layers)
			for out in outs:
				for detection in out:
					scores = detection[5:]
					class_id = np.argmax(scores)
					confidence = scores[class_id]
					if confidence > 0.5:
						centre_x = int(detection[0] * width)
						centre_y = int(detection[1] * height)
						w = int(detection[2] * width)
						h = int(detection[3] * height)
						x = int(centre_x - w / 2)
						y = int(centre_y - h / 2)
						boxes.append([x, y, w, h])
						confidences.append(float(confidence))
						class_ids.append(class_id)
			number_object_detected = len(boxes)
			for i in range(len(boxes)):
				x, y, w, h = boxes[i]
				confidence = str(round(confidences[i], 2))
				label = str(classes[class_ids[i]])

			# if Liveness_Label[np.argmax(score)] == 'training_real_pics' and confidence > 99:
			
			if label == 'Real_Image':
				response_liveness['photo-liveness'] = 'yes'
				response_liveness['confidence'] =confidences
				response_liveness['success'] = True

			else:
				response_liveness['photo-liveness'] = 'no'
				response_liveness['confidence'] = confidences
				response_liveness['success'] = True


			return response_liveness

		except Exception as errormessage:
			
			response_liveness['photo-liveness'] = 'no'
			response_liveness['confidence'] = 0
			response_liveness['success'] = True
			
			return response_liveness




