from deepface import DeepFace
from tests import test_analyze
from tests import test_enforce_detection
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
DeepFace.build_model(model_name="Emotion")
logger.info("Emotion model is just built")

