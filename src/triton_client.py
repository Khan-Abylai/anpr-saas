import tritonclient.grpc as tritongrpcclient
import cv2
import numpy as np
import torch
import time

from qtconsole.qtconsoleapp import flags

from utils import StrLabelConverter
import string
import warnings
warnings.filterwarnings("ignore")

url = '127.0.0.1:8001'

model_name = 'usa_ensemble'
input_name = 'INPUT'
output_name = 'OUTPUT_0'
model_version = '1'
# regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain", "bahrein", "oman", "saudi", "quatar", "kuwait", "others"]

triton_client = tritongrpcclient.InferenceServerClient(url=url, verbose=False)
# model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
# model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

if __name__ == '__main__':

    # ensemble client
    t1 = time.time()
    inputs = []
    outputs = []
    # image_data = np.fromfile("../debug/usa/_ET161D_21-51-19_lp_et161dp_0.972_0.jpeg", dtype="uint8")
    stop =1
    image_data = cv2.imread("../debug/usa/_ET161D_21-51-19_lp_et161dp_0.972_0.jpeg")
    image_data = np.array(cv2.imencode('.jpg', image_data)[1])
    # image_data = np.array(image_data.transpose(2,0,1), dtype=np.uint8).reshape(-1)
    image_data = np.expand_dims(image_data, axis=0)

    inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, 'UINT8'))
    inputs[0].set_data_from_numpy(image_data)

    outputs.append(tritongrpcclient.InferRequestedOutput('LABEL'))
    outputs.append(tritongrpcclient.InferRequestedOutput('PROBABILITY'))
    # outputs.append(tritongrpcclient.InferRequestedOutput('REGION'))

    response = triton_client.infer('usa_ensemble', inputs=inputs, outputs=outputs)

    label = response.as_numpy('LABEL')
    label = np.asarray(label, dtype=str)
    probability = response.as_numpy('PROBABILITY')
    probability = np.asarray(probability, dtype=float)
    # region = response.as_numpy('REGION')
    # region = np.asarray(region, dtype=str)
    print(label, probability)
    print("ensemble exec time: ", time.time() - t1)



    # preprocess, engine, postprocess client
    # t1 = time.time()
    # inputs = []
    # outputs = []
    # image_data = np.fromfile("../debug/mena.png", dtype="uint8")
    # image_data = np.expand_dims(image_data, axis=0)
    # inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, 'UINT8'))
    # inputs[0].set_data_from_numpy(image_data)
    #
    # outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
    # # cls_output = tritongrpcclient.InferRequestedOutput('cls')
    # response = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    #
    # out = response.as_numpy('OUTPUT_0')
    # print("preprocessing time: ", time.time() - t1)
    #
    # t2 = time.time()
    #
    # inputs = []
    # outputs = []
    # inputs.append(tritongrpcclient.InferInput('input', out.shape, 'FP32'))
    # inputs[0].set_data_from_numpy(out)
    # outputs.append(tritongrpcclient.InferRequestedOutput('output'))
    # outputs.append(tritongrpcclient.InferRequestedOutput('cls'))
    # response = triton_client.infer('mena_engine', inputs=inputs, outputs=outputs)
    #
    # out = response.as_numpy('output')
    # cls_out = response.as_numpy('cls')
    # out1 = np.asarray(out, dtype=np.float32)
    # cls_out = np.asarray(cls_out, dtype=np.float32)
    # print("model exec time: ", time.time() - t2)
    #
    # t3 = time.time()
    # inputs = []
    # outputs = []
    # inputs.append(tritongrpcclient.InferInput('input', out.shape, 'FP32'))
    # inputs.append(tritongrpcclient.InferInput('input_cls', cls_out.shape, 'FP32'))
    # inputs[0].set_data_from_numpy(out)
    # inputs[1].set_data_from_numpy(cls_out)
    #
    # outputs.append(tritongrpcclient.InferRequestedOutput('label'))
    # outputs.append(tritongrpcclient.InferRequestedOutput('probability'))
    # outputs.append(tritongrpcclient.InferRequestedOutput('region'))
    # response = triton_client.infer('mena_postprocess', inputs=inputs, outputs=outputs)
    #
    # label = response.as_numpy('label')
    # label = np.asarray(label, dtype=str)
    # probability = response.as_numpy('probability')
    # probability = np.asarray(probability, dtype=float)
    # region = response.as_numpy('region')
    # region = np.asarray(region, dtype=str)
    # print("postprocess exec time: ", time.time() - t3)
    # print("all time: ", time.time() - t1)
    # print(label, probability, region)


    #### engine client

    # img = cv2.imread("../debug/mena.png")
    # w, h = 160, 64
    # x = cv2.resize(img, (w, h))
    # x = x.astype(np.float32) / 255.
    # x = x.transpose(2, 0, 1)
    # x = np.expand_dims(img, axis=0)
    # input = tritongrpcclient.InferInput(input_name, (1, 3, h, w), 'FP32')
    # input.set_data_from_numpy(x)

    # converter = StrLabelConverter(string.digits + string.ascii_lowercase)
    # predictions = torch.from_numpy(out1)
    # cls_predictions = torch.from_numpy(cls_out)
    # cls_idx = cls_predictions.argmax(1).item()
    # predictions = predictions.permute(1, 0, 2).contiguous()
    # prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    # predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    # predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
    # predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
    # predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

    # t2 = time.time()
    # print("triton exec:", t2-t1)
    # print(predicted_test_labels, round(np.mean(predicted_probs), 3), regions[cls_idx])