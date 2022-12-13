# import the necessary packages
import time

import numpy as np
from cv2 import cv2
from scipy import spatial

from input_retrieval import *

# All these classes will be counted as 'vehicles'
list_of_vehicles = ["person", "bicycle", "car", "motorbike", "bus", "truck", "train"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416

# Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, preDefinedConfidence, preDefinedThreshold, USE_GPU = parseCommandLineArguments()
tracking_lines = [
    [548, 768, 540, 0],
    [773, 993, 540, 1],
    [1000, 1220, 540, 1],
]

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
    cv2.putText(
        frame,  # Image
        'Detected Vehicles: ' + str(vehicle_count),  # Label
        (20, 20),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        0.8,  # Size
        (0, 0xFF, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if current_time > start_time:
        os.system('cls')  # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames


# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)


# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
                           (video_width, video_height), True)


# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames,
# the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
# False if the box was not present in the previous frames
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf  # Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height) / 2)):
        return False

    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True


def count_vehicles(idxs_, boxes_, classIDs_, vehicle_count_, vehicle_id_count_, previous_frame_detections_, frame_):
    current_detections_ = {}
    # ensure at least one detection exists
    if len(idxs_) > 0:
        # loop over the indices we are keeping
        for i in idxs_.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes_[i][0], boxes_[i][1])
            (w, h) = (boxes_[i][2], boxes_[i][3])

            centerX_ = x + (w // 2)
            centerY_ = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if LABELS[classIDs_[i]] in list_of_vehicles:
                current_detections_[(centerX_, centerY_)] = vehicle_id_count_
                if not boxInPreviousFrames(previous_frame_detections_, (centerX_, centerY_, w, h), current_detections_):
                    vehicle_id_count_ += 1
                    counted_list.append(0)
                # else: #ID assigning
                # Add the current detection mid-point of box to the list of detected items
                # Get the ID corresponding to the current detection

                ID = current_detections_.get((centerX_, centerY_))
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if list(current_detections_.values()).count(ID) > 1:
                    current_detections_[(centerX_, centerY_)] = vehicle_id_count_
                    vehicle_id_count_ += 1
                    counted_list.append(0)

                for line_ in tracking_lines:
                    if line_[0] < centerX_ < line_[1] and counted_list[ID] == 0:
                        if line_[3]:
                            if centerY_ < line_[2]:
                                vehicle_count_ += 1
                                counted_list[ID] = 1
                        else:
                            if centerY_ > line_[2]:
                                vehicle_count_ += 1
                                counted_list[ID] = 1

                # Display the ID at the center of the box
                color = [0, 0, 255] if counted_list[ID] == 0 else [0, 255, 0]
                cv2.putText(frame_, str(ID), (centerX_, centerY_), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vehicle_count_, vehicle_id_count_, current_detections_


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Using GPU if flag is passed
if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialization
previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]
counted_list = []
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
total_frames, num_frames, vehicle_count, vehicle_id_count = 0, 0, 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
# loop over frames from the video file stream
while True:
    print("================NEW FRAME================")
    num_frames += 1
    total_frames += 1
    print("FRAME:\t", total_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], []

    # Calculating fps each second
    start_time, num_frames = displayFPS(start_time, num_frames)
    # read the next frame from the file
    (grabbed, frame) = videoStream.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Printing the info of the detection
                # print('\nName:\t', LABELS[classID],
                # '\t|\tBOX:\t', x,y)

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    for index, line in enumerate(tracking_lines):
        color = (int(256 / (index + 5)) + 51, int(256 / (index + 1)), int(256 / (index + 3)) + 85)
        cv2.line(frame, (line[0], line[2]), (line[1], line[2]), color, 4)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence, preDefinedThreshold)

    # Draw detection box
    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    vehicle_count, vehicle_id_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count,
                                                                         vehicle_id_count, previous_frame_detections,
                                                                         frame)

    # Display Vehicle Count if a vehicle has passed the line
    displayVehicleCount(frame, vehicle_count)

    # write the output frame to disk
    writer.write(frame)

    if video_width > 1366:
        scale = video_width / 1366
        frame = cv2.resize(frame, (1366, int(video_height // scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Updating with the current frame detections
    previous_frame_detections.pop(0)  # Removing the first frame from the list
    # previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videoStream.release()
