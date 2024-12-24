import openvino as ov
res = ov.convert_model('../triton/models_cpu/usa_detection/1/model.onnx')
ov.serialize(res, "model.xml", "model.bin")