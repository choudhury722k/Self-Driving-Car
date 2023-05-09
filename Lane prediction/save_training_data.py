import cv2
from Predict_steering import process_pipeline
from Camera_callibration import calibrate_camera

def save_image_and_steering_angle(video_file):

    _, mtx, dist, _, _ = calibrate_camera(calib_images_dir='camera_cal')
    curr_steering_angle = 90

    cap = cv2.VideoCapture(video_file)
    try:
        i = 0
        while cap.isOpened():
            return_value, frame = cap.read()

            if return_value:
                pass
            else:
                print('Video has ended or failed, try a different video format!')
                break

            blend, curr_steering_angle = process_pipeline(frame, mtx, dist, curr_steering_angle, keep_state=False)
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, curr_steering_angle), frame)

            i += 1

            cv2.imshow('Frame',blend)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "project_video.mp4"
    # video_path = 0
    save_image_and_steering_angle(video_path)