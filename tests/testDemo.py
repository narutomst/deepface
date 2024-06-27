from PIL import ImageGrab
import time
import cv2
from deepface import DeepFace
from tests import test_analyze
from tests import test_enforce_detection
from deepface.modules import modeling
from deepface.modules import detection
from deepface.modules.modeling import *
from deepface.modules import streaming
from deepface.modules.streaming import *
from deepface.commons import logger as log
logger = log.get_singletonish_logger()

# test_analyze.test_standard_analyze()
# test_analyze.test_analyze_with_all_actions_as_tuple()
# test_analyze.test_analyze_with_all_actions_as_list()
# test_analyze.test_analyze_for_some_actions()
# test_analyze.test_analyze_for_preloaded_image()
# test_analyze.test_analyze_for_different_detectors()


# test_enforce_detection.test_enabled_enforce_detection_for_non_facial_input()
# test_enforce_detection.test_disabled_enforce_detection_for_non_facial_input_on_represent()
# test_enforce_detection.test_disabled_enforce_detection_for_non_facial_input_on_verify()

# 使用DeepFace.stream()执行实时分析功能，但是去掉Age、Gender和人脸对比分析，只要输出Emotion
# 这样就能降低运算量。

# deepface.modules.streaming.analysis()
db_path = "dataset"
model_name = "VGG-Face"
detector_backend = "opencv"
distance_metric = "cosine"

enable_face_analysis = True
enable_face_analysis_Emotion = True
enable_face_analysis_Age = False
enable_face_analysis_Gender = False
enable_face_analysis_Race = False

enable_face_recognition = False
source = 0
time_threshold = 5
frame_threshold = 5
anti_spoofing = False

threshold = 130  # 面部区域的阈值，舍弃较小的阈值
enforce_detection = False

actions = []
if enable_face_analysis_Emotion:
    actions.append("emotion")
if enable_face_analysis_Age:
    actions.append("age")
if enable_face_analysis_Gender:
    actions.append("gender")
if enable_face_analysis_Race:
    actions.append("race")

# build_demography_models(enable_face_analysis=enable_face_analysis)
if enable_face_analysis:
    if enable_face_analysis_Emotion:
        # build_demography_models(enable_face_analysis=enable_face_analysis)
        # DeepFace.build_model(model_name="Emotion")
        # modeling.build_model(model_name="Emotion")
        logger.info("Emotion model is just built")

    if enable_face_analysis_Age:
        # build_demography_models(enable_face_analysis=enable_face_analysis)
        # DeepFace.build_model(model_name="Age")
        modeling.build_model(model_name="Age")
        logger.info("Age model is just built")

    if enable_face_analysis_Gender:
        # build_demography_models(enable_face_analysis=enable_face_analysis)
        # DeepFace.build_model(model_name="Gender")
        modeling.build_model(model_name="Gender")
        logger.info("Gender model is just built")

    if enable_face_analysis_Race:
        # build_demography_models(enable_face_analysis=enable_face_analysis)
        # DeepFace.build_model(model_name="Race")
        modeling.build_model(model_name="Race")
        logger.info("Race model is just built")

if enable_face_recognition:
    # build_facial_recognition_model(model_name=model_name)
    _ = DeepFace.build_model(model_name=model_name)
    logger.info(f"{model_name} is built")

if enable_face_recognition:
    _ = search_identity(
        detected_face=np.zeros([224, 224, 3]),
        db_path=db_path,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        model_name=model_name,
    )

freezed_img = None
freeze = False
num_frames_with_faces = 0
tic = time.time()

# #调试信息准备
# 获取屏幕尺寸
screen_width, screen_height = ImageGrab.grab().size
num_rows = 2
num_cols = 3
area_height = screen_height / num_rows
area_width = screen_width / num_cols

start = True
cap = cv2.VideoCapture(source)  # webcam
while start:
    print("从摄像头读取帧...")
    has_frame, img = cap.read()
    if not has_frame:
        break
    window_name: str = "Fig1 Frame read"
    cv2.imshow(window_name, img)
    cv2.moveWindow(window_name, 0, 0)
    # cv2.moveWindow()函数的参数是窗口的新位置，这个位置是相对于屏幕左上角的坐标。所以，当你的参数是(0, 0)
    # 时，窗口的左上角就会被移动到屏幕的左上角。

    # we are adding some figures into img such as identified facial image, age, gender
    # that is why, we need raw image itself to make analysis
    # 我们要在图像img中添加一些数据，例如已识别的面部图像、年龄、性别，这就是为什么我们需要原始图像本身来进行分析。
    raw_img = img.copy()  # 原始图像

    faces_coordinates = []
    if freeze is False:
        # faces_coordinates = grab_facial_areas(
        #     img=img, detector_backend=detector_backend, anti_spoofing=anti_spoofing
        # )
        try:
            face_objs = detection.extract_faces(
                img_path=img,
                detector_backend=detector_backend,
                # you may consider to extract with larger expanding value
                expand_percentage=0,
                anti_spoofing=anti_spoofing,
            )
            faces = [
                (
                    face_obj["facial_area"]["x"],
                    face_obj["facial_area"]["y"],
                    face_obj["facial_area"]["w"],
                    face_obj["facial_area"]["h"],
                    face_obj.get("is_real", True),
                    face_obj.get("antispoof_score", 0),
                )
                for face_obj in face_objs
                if face_obj["facial_area"]["w"] > threshold
            ]
            faces_coordinates = faces
            print("人脸坐标获取成功")
        except:  # to avoid exception if no face detected
            print("no face detected, continue")
            #continue

        # we will pass img to analyze modules (identity, demography) and add some illustrations
        # that is why, we will not be able to extract detected face from img clearly
        # 我们将把 img 传递给分析模块（身份、人口），并添加一些插图。这就是为什么我们无法从 img 中清晰提取检测到的人脸的原因。
        detected_faces = extract_facial_areas(img=img, faces_coordinates=faces_coordinates)
        # deepface.modules.streaming.extract_facial_areas()具体实现代码如下：
        # detected_faces = []
        # for x, y, w, h, is_real, antispoof_score in faces_coordinates:
        #     detected_face = img[int(y): int(y + h), int(x): int(x + w)]
        #     detected_faces.append(detected_face)

        img = highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)
        # cv2.imshow("Fig2 highlight_facial_areas() output image", img)
        window_name: str = "Fig2 highlight_facial_areas() output image"
        cv2.imshow(window_name, img)
        cv2.moveWindow(window_name, int(area_width), 0)

        # 这里我们使用了一个名为countdown_to_freeze()的函数来计算帧数，并判断是否达到阈值。
        img = countdown_to_freeze(
            img=img,
            faces_coordinates=faces_coordinates,
            frame_threshold=frame_threshold,
            num_frames_with_faces=num_frames_with_faces,
        )
        # cv2.imshow("Fig3 countdown_to_freeze() output image", img)
        window_name: str = "Fig3 countdown_to_freeze() output image"
        cv2.imshow(window_name, img)
        cv2.moveWindow(window_name, int(area_width*2), 0)

        num_frames_with_faces = num_frames_with_faces + 1 if len(faces_coordinates) else 0
        print("num_frames_with_faces:", num_frames_with_faces)
        freeze = num_frames_with_faces > 0 and num_frames_with_faces % frame_threshold == 0
        print("freeze:", freeze)

        if freeze:
            # add analyze results into img - derive from raw_img
            img = highlight_facial_areas(
                img=raw_img, faces_coordinates=faces_coordinates, anti_spoofing=anti_spoofing
            )
            # cv2.imshow("Fig4 highlight_facial_areas()output image", img)
            window_name: str = "Fig4 highlight_facial_areas()output image"
            cv2.imshow(window_name, img)
            cv2.moveWindow(window_name, 0, int(area_height))
            # age, gender and emotion analysis
            # img = perform_demography_analysis(
            #     enable_face_analysis=enable_face_analysis,
            #     img=raw_img,
            #     faces_coordinates=faces_coordinates,
            #     detected_faces=detected_faces,
            # )
            if enable_face_analysis is False:
                # return img
                break
            for idx, (x, y, w, h, is_real, antispoof_score) in enumerate(faces_coordinates):
                detected_face = detected_faces[idx]
                # analyze()函数既可以接受原始图像（含一张或多张人脸）数组作为输入，此时需要从图中查找人脸，因此需指定detector_bakend
                # 也可以接受检测到的人脸图像数组作为输入，此时无需从图中查找人脸，因此detector_backend参数指定为skip。
                # 下面的用法就是后一种情况
                demographies = DeepFace.analyze(
                    img_path=detected_face,
                    actions=actions,
                    detector_backend="skip",
                    enforce_detection=enforce_detection,
                    silent=False,
                )

                if len(demographies) == 0:
                    continue

                # safe to access 1st index because detector backend is skip
                demography = demographies[0]

                img = overlay_emotion(img=raw_img, emotion_probas=demography["emotion"], x=x, y=y, w=w, h=h)
                # cv2.imshow("Fig5 overlay_emotion()output image", img)
                window_name: str = "Fig5 overlay_emotion()output image"
                cv2.imshow(window_name, img)
                cv2.moveWindow(window_name, int(area_width), int(area_height))

                if enable_face_analysis_Age and enable_face_recognition:
                    img = overlay_age_gender(
                        img=img,
                        apparent_age=demography["age"],
                        gender=demography["dominant_gender"][0:1],  # M or W
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                    )

            # facial recognition analysis
            if enable_face_recognition:
                img = perform_facial_recognition(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                    db_path=db_path,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    model_name=model_name,
                )

            # freeze the img after analysis
            freezed_img = img.copy()

            # start counter for freezing
            tic = time.time()
            logger.info("freezed")

    elif freeze is True and time.time() - tic > time_threshold:
        freeze = False
        freezed_img = None
        # reset counter for freezing
        tic = time.time()
        logger.info("freeze released")

    freezed_img = countdown_to_release(img=freezed_img, tic=tic, time_threshold=time_threshold)
    window_name = "img" if freezed_img is None else "freezed_img"
    cv2.imshow(window_name, img if freezed_img is None else freezed_img)
    cv2.moveWindow(window_name, int(area_width*2), int(area_height))

    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break
    # 测试开关
    start = False

print("循环运行完毕")
print("按q键退出，或者倒计时结束自动退出")
# 设置倒计时的秒数
countdown_seconds = 3600

# 获取当前时间
start_time = time.time()

while True:
    # 计算剩余时间
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = countdown_seconds - int(elapsed_time)

    # 如果倒计时结束，退出循环
    if remaining_time <= 0:
        break

    # 显示倒计时
    print(remaining_time)

    # 暂停1秒
    time.sleep(10)
# # kill open cv things
# cap.release()
# cv2.destroyAllWindows()
