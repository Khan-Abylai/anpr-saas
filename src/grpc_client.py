import cv2
import numpy as np
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import InferenceServerException
from utils import DetectionUtils
import warnings
warnings.filterwarnings("ignore")

class DetectionTriton(object):
    def __init__(self, triton_client, model_name, model_version='1', img_width=512, img_height=512, nms_threshold=0.45, conf_threshold=0.7):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = 'actual_input'
        self.output_name = 'output'
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.triton_client = triton_client
        self.model_metadata = self.triton_client.get_model_metadata(model_name=self.model_name,
                                                                    model_version=self.model_version)
        self.model_config = self.triton_client.get_model_config(model_name=self.model_name,
                                                                model_version=self.model_version)
        self.detection_utils = DetectionUtils(img_width=img_width, img_height=img_height, nms_thres=self.nms_threshold,
                                        conf_thres=self.conf_threshold)


    def detect(self, image, draw=True, squared=False):
        img_orig = image.copy()
        inputs = []
        outputs = []
        image_model = self.detection_utils.preprocess(img_orig)
        image_data = np.expand_dims(image_model, axis=0)
        inputs.append(tritongrpcclient.InferInput(self.input_name, image_data.shape, 'FP32'))
        inputs[0].set_data_from_numpy(image_data)
        outputs.append(tritongrpcclient.InferRequestedOutput(self.output_name))
        response = self.triton_client.infer(self.model_name, inputs=inputs, outputs=outputs)
        predictions = response.as_numpy('output')
        drawn_image, plate_images, flag = self.detection_utils.postprocess(predictions[0], img_orig, draw=draw, squared=squared)

        return drawn_image, plate_images, flag

class RecognitionTriton(object):
    def __init__(self, triton_client, model_name, model_version='1'):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = 'INPUT'
        self.output_name = 'OUTPUT_0'
        self.triton_client = triton_client
        self.model_metadata = self.triton_client.get_model_metadata(model_name=self.model_name,
                                                                    model_version=self.model_version)
        self.model_config = self.triton_client.get_model_config(model_name=self.model_name,
                                                                model_version=self.model_version)


    def recognize(self, plate_images):
        labels = []
        probs = []
        for plate_image in plate_images:
            inputs = []
            outputs = []
            image = plate_image.copy()
            image_data = np.array(cv2.imencode('.jpg', image)[1])
            image_data = np.expand_dims(image_data, axis=0)
            inputs.append(tritongrpcclient.InferInput(self.input_name, image_data.shape, 'UINT8'))
            inputs[0].set_data_from_numpy(image_data)

            outputs.append(tritongrpcclient.InferRequestedOutput('LABEL'))
            outputs.append(tritongrpcclient.InferRequestedOutput('PROBABILITY'))
            # outputs.append(tritongrpcclient.InferRequestedOutput('REGION'))

            response = self.triton_client.infer('usa_ensemble', inputs=inputs, outputs=outputs)

            label = response.as_numpy('LABEL')
            label = np.asarray(label, dtype=str)
            probability = response.as_numpy('PROBABILITY')
            probability = np.asarray(probability, dtype=float)
            # region = response.as_numpy('REGION')
            # region = np.asarray(region, dtype=str)
            labels.append(label)
            probs.append(probability)

        return labels, probs



# test detector
# url = '127.0.0.1:8001'
#
# model_name = 'usa_detection'
# input_name = 'actual_input'
# output_name = 'output'
# model_version = '1'
#
# detection = DetectionUtils(img_width=512, img_height=512, nms_thres=0.45, conf_thres=0.7)
# triton_client = tritongrpcclient.InferenceServerClient(url=url, verbose=False)
# model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
# model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)
#
# if __name__ == '__main__':
#
#     # detection client
#     t1 = time.time()
#     inputs = []
#     outputs = []
#
#     image = cv2.imread("../debug/usa2.jpeg")
#     image_model = detection.preprocess(image)
#     image_data = np.expand_dims(image_model, axis=0)
#     inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, 'FP32'))
#     inputs[0].set_data_from_numpy(image_data)
#
#     outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
#
#     response = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
#
#     predictions = response.as_numpy('output')
#
#     plates = detection.postprocess(predictions[0], image)
#
#     for idx, plate in enumerate(plates):
#         cv2.imwrite("../debug/usa_plate_" + str(idx) + ".jpg", plate)



