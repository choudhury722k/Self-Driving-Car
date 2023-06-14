import os
import os.path as path
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from BirdEye import birdeye
from Binarizing import binarize
from Camera_callibration import calibrate_camera, undistort
from Lanelines import Line, get_fits_by_sliding_windows, get_fits_by_previous_fits, draw_back_onto_the_road

ym_per_pix = 30 / 720   # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

time_window = 10        # results are averaged over this number of frames

processed_frames = 0                    # counter of frames processed (when processing video)
line_lt = Line(buffer_len=time_window)  # line on the left of the lane
line_rt = Line(buffer_len=time_window)  # line on the right of the lane

def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter, steering_angle):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 50), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 50), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Steering angle: {:.02f}deg'.format(steering_angle), (860, 100), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road

def compute_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.
    """
    if line_lt.detected and line_rt.detected:
        line_lt_bottom_x = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom_x = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom_x - line_lt_bottom_x
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom_x + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter

def compute_steering_angle_from_center(line_lt, line_rt, frame_width, frame_height):
    if line_lt.detected and line_lt.detected:
        left_x2 = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        right_x2 = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])

    # if line_lt.detected == "True" and line_lt.detected == "False":
    #     left_x2 = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
    #     right_x2 = line_lt.all_x.max()

    # if line_lt.detected == "False" and line_lt.detected == "True":
    #     left_x2 = line_rt.all_x.min()
    #     right_x2 = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
    
    mid = int(frame_width / 2)
    x_offset = (left_x2 + right_x2) / 2 - mid
    y_offset = int(frame_height / 2)
    # print(x_offset)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel
    # print(steering_angle)

    return steering_angle

def display_heading_line(frame, steering_angle, line_color=(255, 255, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, 
                             max_angle_deviation_two_lines=5, 
                             max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    if new angle is too different from current angle, 
    only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
            + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    return stabilized_steering_angle

def process_pipeline(frame, mtx, dist, steering_angle, keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    """
    global line_lt, line_rt, processed_frames, curr_steering_angle
    curr_steering_angle = steering_angle

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    # compute steering_angle in degree from center of the lane
    new_steering_angle = compute_steering_angle_from_center(line_lt, line_rt, frame_width=frame.shape[1], frame_height=frame.shape[0])

    # 
    if line_lt.detected and line_lt.detected:
        lane_lines = 2
    else:
        lane_lines = 1
    curr_steering_angle = stabilize_steering_angle(curr_steering_angle, new_steering_angle, lane_lines)

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter, curr_steering_angle)

    # draw a guiding line
    final_image = display_heading_line(blend_output, curr_steering_angle)

    processed_frames += 1

    return final_image, curr_steering_angle

if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
    steering_angle = 90
    
    mode = 'video'
    
    if mode == 'video':
        video_path = r"/home/soumya/Self-Driving-Car/Lane_Prediction/project_video.mp4"
        # video_path = 0

        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        while True:
            return_value, frame = vid.read()
            if return_value:
                pass
            else:
                print('Video has ended or failed, try a different video format!')
                break

            blend, steering_angle = process_pipeline(frame, mtx, dist, steering_angle, keep_state=False)
            cv2.imshow('Frame',blend)
            if cv2.waitKey(25) & 0xFF == ord('q'): break
        
        vid.release()
        cv2.destroyAllWindows()
    else:
        # test_img_dir = 'test_images'
        # for test_img in os.listdir(test_img_dir):
        #     frame = cv2.imread(os.path.join(test_img_dir, test_img))
        #     blend = process_pipeline(frame, keep_state=False)
        #     cv2.imwrite('output_images/{}'.format(test_img), blend)
        #     plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
        #     plt.show()

        test_img_dir = r"/home/soumya/Self-Driving-Car/Lane_Prediction/test_images/test1.jpg"
        frame = cv2.imread(test_img_dir)
        blend = process_pipeline(frame, mtx, dist, steering_angle, keep_state=False)
        plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
        plt.show()