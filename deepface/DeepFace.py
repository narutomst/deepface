# common dependencies
import os
import warnings
import logging
from typing import Any, Dict, List, Union, Optional

# this has to be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# pylint: disable=wrong-import-position

# 3rd party dependencies
import numpy as np
import pandas as pd
import tensorflow as tf

# package dependencies
from deepface.commons import package_utils, folder_utils
from deepface.commons import logger as log
from deepface.modules import (
    modeling,
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
    preprocessing,
)
from deepface import __version__

logger = log.get_singletonish_logger()

# -----------------------------------
# configurations for dependencies

# users should install tf_keras package if they are using tf 2.16 or later versions
# 如果用户使用的是 tf 2.16 或更高版本，则应安装 tf_keras 软件包
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

# create required folders if necessary to store model weights
folder_utils.initialize_folder()


def build_model(model_name: str) -> Any:
    """
    This function builds a deepface model
    Args:
        model_name (string): face recognition or facial attribute model      人脸识别或人脸属性模型
            VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
            Age, Gender, Emotion, Race for facial attributes
            用于人脸识别的 VGG-Face、Facenet、OpenFace、DeepFace、DeepID
            用于面部属性的年龄Age、性别Gender、情感Emotion、种族Race
    Returns:
        built_model
    """
    return modeling.build_model(model_name=model_name)


def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons. 验证图像对代表的是同一个人还是不同的人。

    Args:
        img1_path (str or np.ndarray or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images or pre-calculated embeddings. 预先计算的嵌入

        img2_path (str or np.ndarray or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images or pre-calculated embeddings. 预先计算的嵌入

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv). 探测器后台

        distance_metric (string): Metric for measuring similarity.  衡量相似性的指标
            Options: 'cosine', 'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).
        如果在图像中未检测到人脸，则引发异常。设置为 "假 "可避免低分辨率图像出现异常
        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).
            按百分比扩大检测到的面部区域
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)
            在将输入图像输入模型之前，对其进行归一化处理。
        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).
            抑制或允许某些日志信息，使分析过程更安静
        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).
            指定一个阈值，以确定数据对代表的是同一个人还是不同的个体。该阈值用于比较距离。如果不设置，将根据指定的模型名称和距离度量（默认为无）应用默认的预调阈值。
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False). 启用防欺骗标志

    Returns:
        result (dict): A dictionary containing verification results with following keys.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity. 距离越小，表示相似度越高

        - 'threshold' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.
            如果距离低于这个阈值，则认为图像是匹配的。
        - 'model' (str): The chosen face recognition model.

        - 'distance_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images. 两幅图像中人脸的矩形感兴趣区域。
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds. 验证过程耗时（秒）
    """

    return verification.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        silent=silent,
        threshold=threshold,
        anti_spoofing=anti_spoofing,
    )


def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.
    分析所提供图像中的面部特征，如年龄、性别、情绪和种族。
    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.
            图片的确切路径，BGR 格式的 numpy 数组、或 base64 编码的图像。如果源图像包含多个人脸，结果将 包括每个检测到的面孔的信息。
        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.
            要分析的属性。默认为（"年龄"、"性别"、"情感"、"种族"）。如果需要，您可以将其中一些属性排除在分析之外。
        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set False to avoid the exception for low-resolution images (default is True).
            如果在图像中没有检测到人脸，则会引发异常。设置为 "False"可避免低分辨率图像出现异常（默认为 "True"）。
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv). 探测器后台

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
           the analysis results for a detected face. Each dictionary in the list contains the
           following keys:

        - 'region' (dict): Represents the rectangular region of the detected face in the image.代表图像中检测到的人脸的矩形区域。
            - 'x': x-coordinate of the top-left corner of the face.
            - 'y': y-coordinate of the top-left corner of the face.
            - 'w': Width of the detected face region.
            - 'h': Height of the detected face region.

        - 'age' (float): Estimated age of the detected face.

        - 'face_confidence' (float): Confidence score for the detected face.
            Indicates the reliability of the face detection. 检测到的人脸的可信度得分。表示人脸检测的可靠性

        - 'dominant_gender' (str): The dominant gender in the detected face.
            Either "Man" or "Woman". 检测脸部的主导性别。或是"Man" or "Woman"

        - 'gender' (dict): Confidence scores for each gender category. 各性别类别的置信度得分
            - 'Man': Confidence score for the male gender.
            - 'Woman': Confidence score for the female gender.

        - 'dominant_emotion' (str): The dominant emotion in the detected face.
            Possible values include "sad","angry","surprise","fear","happy",
            "disgust" and "neutral"
            检测到的人脸中的主要情绪。可能的值包括 "悲伤"、"愤怒"、"惊讶"、"恐惧"、"快乐"、"厌恶 "和 "中性"。
        - 'emotion' (dict): Confidence scores for each emotion category. 每个情绪类别的置信度分数
            - 'sad': Confidence score for sadness.
            - 'angry': Confidence score for anger.
            - 'surprise': Confidence score for surprise.
            - 'fear': Confidence score for fear.
            - 'happy': Confidence score for happiness.
            - 'disgust': Confidence score for disgust.
            - 'neutral': Confidence score for neutrality.

        - 'dominant_race' (str): The dominant race in the detected face.  检测脸部的主要种族。
            Possible values include "indian","asian","latino hispanic",
            "black","middle eastern" and "white".
            可能的值包括 "印度人"、"亚洲人"、"拉丁裔西班牙人"、"黑人"、"中东人 "和 "白人"。
        - 'race' (dict): Confidence scores for each race category. 每个种族类别的置信度得分
            - 'indian': Confidence score for Indian ethnicity.
            - 'asian': Confidence score for Asian ethnicity.
            - 'latino hispanic': Confidence score for Latino/Hispanic ethnicity.
            - 'black': Confidence score for Black ethnicity.
            - 'middle eastern': Confidence score for Middle Eastern ethnicity.
            - 'white': Confidence score for White ethnicity.
    """
    return demography.analyze(
        img_path=img_path,
        actions=actions,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        silent=silent,
        anti_spoofing=anti_spoofing,
    )


def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
    refresh_database: bool = True,
    anti_spoofing: bool = False,
) -> List[pd.DataFrame]:
    """
    Identify individuals in a database   识别数据库中的个人
    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.
            图片的确切路径，BGR 格式的 numpy 数组、 或 base64 编码的图像。如果源图像包含多个人脸，结果将 包括每个检测到的面孔的信息。
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.
            包含图像文件的文件夹路径。数据库中所有检测到的面孔 数据库中所有检测到的面孔都将在决策过程中予以考虑。
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv). 探测器后台

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).
            指定一个阈值，以确定数据对代表的是同一个人还是不同的个体。该阈值用于比较距离。如果不设置，将根据指定的模型名称和距离度量应用默认的预调阈值
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        refresh_database (boolean): Synchronizes the images representation (pkl) file with the
            directory/db files, if set to false, it will ignore any file changes inside the db_path
            (default is True).
            将图像表示法 (pkl) 文件与目录/数据库文件同步，如果设置为 false，将忽略 db_path 内的任何文件更改（默认为 True）。
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[pd.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
            to the identity information for an individual detected in the source image.
            The DataFrame columns include:
            pandas数据帧列表。每个数据帧都与源图像中检测到的个体的身份信息相对应。DataFrame列包括
        - 'identity': Identity label of the detected individual. 被检测者的身份标签

        - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                target face in the database. 数据库中目标人脸的边界框坐标。

        - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                detected face in the source image. 源图像中检测到的人脸的边界框坐标。

        - 'threshold': threshold to determine a pair whether same person or different persons 确定一对（图像）是同一人还是不同人的阈值

        - 'distance': Similarity score between the faces based on the
                specified model and distance metric
    """
    return recognition.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
        refresh_database=refresh_database,
        anti_spoofing=anti_spoofing,
    )


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings. 用多维向量嵌入表示面部图像

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
            (default is VGG-Face.).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set as False to avoid the exception for low-resolution images
            (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
            (default is base).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.代表面部特征的多维向量。
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
            维数因参考模型而异（例如，FaceNet 返回 128 维，VGG-Face 返回 4096 维）。

        - facial_area (dict): Detected facial area by face detection in dictionary format.人脸检测(算法)检测到的面部区域（字典格式）
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.
            包含作为左角点的 "x "和 "y"，以及作为宽度和高度的 "w "和 "h"。如果 "detector_backend "设置为 "skip"，则表示整个图像区域，是无意义的。
        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
            人脸检测的置信度得分。如果 "detector_backend "设置为 "skip"，置信度将为 0，且毫无意义。
    """
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
    )


def stream(
    db_path: str = "",
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enable_face_analysis: bool = True,
    source: Any = 0,
    time_threshold: int = 5,
    frame_threshold: int = 5,
    anti_spoofing: bool = False,
) -> None:
    """
    Run real time face recognition and facial attribute analysis
    运行实时人脸识别和人脸属性分析
    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.
            包含图像文件的文件夹路径。决策过程将考虑数据库中所有检测到的面孔。
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enable_face_analysis (bool): Flag to enable face analysis (default is True).

        source (Any): The source for the video stream (default is 0, which represents the
            default camera). 视频流的来源（默认为 0，表示默认摄像机），OBS虚拟摄像头貌似是1

        time_threshold (int): The time threshold (in seconds) for face recognition (default is 5). 人脸识别的时间阈值（秒）（默认为 5）

        frame_threshold (int): The frame threshold for face recognition (default is 5). 人脸识别的帧阈值（默认为 5）

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
    Returns:
        None
    """

    time_threshold = max(time_threshold, 1)
    frame_threshold = max(frame_threshold, 1)

    streaming.analysis(
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enable_face_analysis=enable_face_analysis,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
        anti_spoofing=anti_spoofing,
    )


def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image
    从给定图像中提取人脸
    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        grayscale (boolean): Flag to convert the image to grayscale before
            processing (default is False). 处理前将图像转换为灰度的标志（默认为 False）

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values. left and right eyes
                are eyes on the left and right respectively with respect to the person itself
                instead of observer.
                键 "left_eye"（左眼）和 "right_eye"（右眼）的值是由 2 个 int 组成的元组。
                左眼和右眼分别指相对于人本身的左眼和右眼，而不是观察者。
        - "confidence" (float): The confidence score associated with the detected face.

        - "is_real" (boolean): antispoofing analyze result. this key is just available in the
            result only if anti_spoofing is set to True in input arguments.
            反欺骗分析结果。只有在输入参数中将 anti_spoofing 设置为 True 时，结果中才会出现此键。
        - "antispoof_score" (float): score of antispoofing analyze result. this key is
            just available in the result only if anti_spoofing is set to True in input arguments.
            反欺骗分析结果的得分。只有在输入参数中将 anti_spoofing 设置为 True 时，结果中才会出现此键。
    """

    return detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        grayscale=grayscale,
        anti_spoofing=anti_spoofing,
    )


def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()


# deprecated function(s)


def detectFace(
    img_path: Union[str, np.ndarray],
    target_size: tuple = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> Union[np.ndarray, None]:
    """
    Deprecated face detection function. Use extract_faces for same functionality.
    已废弃的人脸检测功能。使用 extract_faces 可实现相同功能。
    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image (default is (224, 224)).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

    Returns:
        img (np.ndarray): detected (and aligned) facial area image as numpy array
    """
    logger.warn("Function detectFace is deprecated. Use extract_faces instead.")
    face_objs = extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        grayscale=False,
    )
    extracted_face = None
    if len(face_objs) > 0:
        extracted_face = face_objs[0]["face"]
        extracted_face = preprocessing.resize_image(img=extracted_face, target_size=target_size)
    return extracted_face
