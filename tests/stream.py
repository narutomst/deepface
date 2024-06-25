from deepface import DeepFace
# GPU显存4G运行时直接报错，建议尝试显存6G以上的显卡。或者使用CPU版本tensorflow

# DeepFace.stream("dataset", enable_face_analysis=False, anti_spoofing=True)  # opencv
# DeepFace.stream("dataset", enable_face_analysis=True, anti_spoofing=True)  # opencv
# 一直在输出提示信息
# 24-06-23 16:52:45 - Downloading MiniFASNetV2 weights to C:\Users\Administrator/.deepface/weights/2.7_80x80_MiniFASNetV2.pth

DeepFace.stream("dataset", detector_backend = 'opencv')
# DeepFace.stream("dataset", detector_backend = 'ssd')
# DeepFace.stream("dataset", detector_backend = 'mtcnn')
# DeepFace.stream("dataset", detector_backend = 'dlib')
# DeepFace.stream("dataset", detector_backend = 'retinaface')
