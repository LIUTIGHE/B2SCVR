import sys
import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2
import subprocess

from mvextractor.videocap import VideoCap

def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        shift = 2
        factor = (1 << shift)
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (int((mv[0, 5] + mv[0, 7] / mv[0, 9]) * factor + 0.5), int((mv[0, 6] + mv[0, 8] / mv[0, 9]) * factor + 0.5))
            end_pt = (mv[0, 5] * factor, mv[0, 6] * factor)
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, shift, 0.1)
    return frame

def draw_mv_flow(frame, motion_vectors):
    h, w = frame.shape[:2]
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        shift = 2
        factor = (1 << shift)
        scale = 100     # magnitude vector for enhanced visualization
        
        for mv in motion_vectors:
            mb_w, mb_h = mv[1], mv[2]  # get macroblock size
            dst_x, dst_y = mv[5], mv[6]  # motion vector start point (block position in current frame)
            motion_x, motion_y = mv[7] / mv[9], mv[8] / mv[9]  # calculate actual displacement (considering motion_scale)
            
            dx = int(motion_x * factor * scale)
            dy = int(motion_y * factor * scale)
            
            # draw macroblock border
            x_start, y_start = dst_x, dst_y
            x_end, y_end = x_start + mb_w, y_start + mb_h
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)
            
            # draw motion vector arrow
            end_x, end_y = dst_x + dx, dst_y + dy
            cv2.arrowedLine(frame, (dst_x, dst_y), (end_x, end_y), (0, 255, 0), 1, cv2.LINE_AA)
            
            # fill flow_map
            flow_map[y_start:y_end, x_start:x_end, 0] = dx
            flow_map[y_start:y_end, x_start:x_end, 1] = dy
    
    # Convert flow to HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    magnitude, angle = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue: direction
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value: magnitude
    
    optical_flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return optical_flow_img

argument_parser = argparse.ArgumentParser(description=__doc__)
argument_parser.add_argument('--input', '-i', type=str, required=True, help="Input bitstream file path.")
args = argument_parser.parse_args()

def main():
    # first extract frames from the input video using ffmpeg
    bitstream_dir = args.input
    bsc_img = bitstream_dir.split("/")[0] + "/bsc_imgs/"
    if not os.path.exists(bsc_img):
        os.mkdir(bsc_img)
    mv_map = bitstream_dir.split("/")[0] + "/mv_maps/"
    if not os.path.exists(mv_map):
        os.mkdir(mv_map)
    frame_type_dir = bitstream_dir.split("/")[0] + "/frame_types/"
    if not os.path.exists(frame_type_dir):
        os.mkdir(frame_type_dir)
        
    frame_dir = bitstream_dir.split("/")[0] + "/bsc_imgs/" + bitstream_dir.split("/")[1][:-7]
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
    cmd = f"ffmpeg -i {bitstream_dir} -start_number 0 -qscale:v 2 {frame_dir}/%05d.jpg"
    subprocess.run(cmd, shell=True)
    
    cap = VideoCap()
    ret = cap.open(bitstream_dir)
    if not ret:
        raise RuntimeError(f"Could not open {bitstream_dir}")
    else:
        print(f"Successfully open {bitstream_dir}.")

    step = 0
    times = []
    frame_num = len(os.listdir(frame_dir))

    while step < frame_num:
        print("Frame: ", step, end=" ")
        
        tstart = time.perf_counter()
        
        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        tend = time.perf_counter()
        telapsed = tend - tstart
        times.append(telapsed)

        # if not ret:
        #     print("No frame read. Stopping.")
        #     break

        print("timestamp: {} | ".format(timestamp), end=" ")
        print("frame type: {} | ".format(frame_type), end=" ")
        print("frame size: {} | ".format(np.shape(frame)), end=" ")
        print("motion vectors: {} | ".format(np.shape(motion_vectors)), end=" ")
        print("elapsed time: {} s".format(telapsed), end="\r")

        # frame = draw_motion_vectors(frame, motion_vectors)
        frame = draw_mv_flow(frame, motion_vectors)

        mv_dumpdir = bitstream_dir.split("/")[0] + "/mv_maps/" + bitstream_dir.split("/")[1]
        if not os.path.exists(mv_dumpdir):
            os.mkdir(mv_dumpdir)
        cv2.imwrite(os.path.join(mv_dumpdir, f"{step:05d}.jpg"), frame)
            
        pm_dumpdir = bitstream_dir.split("/")[0] + "/frame_types/" + bitstream_dir.split("/")[1] + ".txt"
        with open(pm_dumpdir, "a") as f:
            f.write(frame_type+"\n")
            
        step += 1

    cap.release()
    
    # check the subfolders in the frame_dir and create a test_list.txt file from scratch
    list_file = bitstream_dir.split("/")[0] + "/test_list.txt"
    with open(list_file, "w") as f:
        # write the video name
        f.write(bitstream_dir.split("/")[1][:-7] + "\n")

if __name__ == "__main__":
    main()
    