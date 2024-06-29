import cv2
import os
import numpy as np


def extract_frames(video_path, output_dir, num_frames=4):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    work_frames = 100
    frame_count=0
    ret , first_frame = cap.read()
    base_frame =np.array(first_frame)

    def frame_difference(frame1, frame2):
        # Compute absolute difference between the two frames
        diff = cv2.absdiff(frame1, frame2)
        # Convert the difference image to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Sum the pixel values to get a measure of the difference
        diff_sum = np.sum(gray_diff)
        return diff_sum

    for frame_idx in range(work_frames):
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Process the frame (for example, display it)
        frame = np.array(frame)
        frame_count += 1
        # get frames for extraction
        base_frame = np.median([frame, base_frame], axis=0).astype(np.uint8)
        # print(frame_idx)


        # Wait for a key event for a short period (1 ms) and exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imshow('base', base_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # diffList =[];
    max_diff = 0
    max_diff_frame = None
    max_diff_index = -1
    for frame_idx in range(work_frames):
        ret, frame = cap.read()
        if not ret:
            break
        diff = frame_difference(base_frame, frame)
        if diff > max_diff:
            max_diff = diff
            max_diff_frame = frame
            max_diff_index = frame_idx
    if max_diff_frame is not None:
        cv2.imshow('Frame with Maximum Difference', max_diff_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(
            f'The frame with the maximum difference is frame_{max_diff_index + 1}.jpg with a difference of {max_diff}')
    else:
        print('No frames were compared.')

# Example usage
video_path = '' #add video path here
output_dir = 'extracted_frames'
extract_frames(video_path, output_dir)


