import cv2

# Open the video
video_capture = cv2.VideoCapture('video.mp4')

# Read the first frame
success, prev_frame = video_capture.read()

# Read the frames one by one
frame_number = 0
success, frame = video_capture.read()
while success:
    # Compute the optical flow between the previous frame and the current frame
    flow = cv2.calcOpticalFlowPyrLK(prev_frame, frame, None, None, None, None, (10, 10))

    # Select keyframes based on motion
    motion_sum = sum(flow)
    if motion_sum > THRESHOLD:
        cv2.imwrite('keyframe_{}.jpg'.format(frame_number), frame)

    # Save the current frame as the previous frame for the next iteration
    prev_frame = frame

    # Read the next frame
    frame_number += 1
    success, frame = video_capture.read()
video_capture.release()
