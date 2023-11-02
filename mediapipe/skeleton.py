import mediapipe as mp
import numpy as np
    
import cv2

cam = cv2.VideoCapture("20230905 copy.avi")
mppose = mp.solutions.pose
mpdraw = mp.solutions.drawing_utils
poses = mppose.Pose()
h = 0
w = 0

throw = False

def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]

def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    m_hip = (r_hip + l_hip)
    m_hip = [x / 2 for x in m_hip]
    m_knee = (r_knee + l_knee)
    m_knee = [x / 2 for x in m_knee]
    m_ankle = (r_ankle + l_ankle)
    m_ankle = [x / 2 for x in m_ankle]

    mid_angle = calc_angles(m_hip, m_knee, m_ankle)

    return [int(r_angle),int(l_angle) ,int(mid_angle)]

model_path = '/absolute/path/to/pose_landmarker_full.task'


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# # Create a pose landmarker instance with the video mode:
# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.VIDEO)

def main():
    global h,w,status

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:
        ret , frame = cam.read()

        if not ret:
            break

        rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poseoutput = poses.process(rgbframe)
        h, w, _ = frame.shape
        preview = frame.copy()

        if poseoutput.pose_landmarks:
            mpdraw.draw_landmarks(preview, poseoutput.pose_landmarks, mppose.POSE_CONNECTIONS)
            knee_angles = get_knee_angle(poseoutput.pose_landmarks.landmark)


            throwing_leg_angle = knee_angles[0]

            if throwing_leg_angle >= 175:
                throw = False        
            elif throwing_leg_angle <175:
                throw = True

            if throw:
                cv2.imshow('throw',preview)
                cv2.waitkey(4)
            else:
                cv2.imshow('throw',preview)
                cv2.waitkey(1)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

            
