import cv2

#for coursework, 2 tagged objetcs are needed
#using a predefined dataset
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def detect_markers(frame):
    corners, ids, rejected = detector.detectMarkers(frame)
    return corners, ids