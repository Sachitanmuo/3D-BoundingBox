import torch
from TRT import TRT
import time
import onnx
import onnxruntime as ort

def main():
    model_path = 'C:/Users/99/Desktop/GitHub/3D-BoundingBox/trt/trt/epoch_10.engine'
    onnx_path = "./epoch_10.onnx"
    trt_model = TRT(model_path=model_path, fp16=False)
    trt_model.start()

    # 載入 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    ort_session = ort.InferenceSession(onnx_path)

    for i in range(10):
        input_shape = (1, 3, 224, 224)
        input_tensor = torch.randn(input_shape).to('cuda')
        start_time = time.time()
        output_trt = trt_model.predict(input_tensor)
        end_time = time.time()

        conf_trt, dim_trt, orient_trt = output_trt

        orient_trt = orient_trt[0, :]
        print(f"TensorRT orient: {orient_trt}")
        conf_trt = conf_trt[0, :]
        print(f"TensorRT conf: {conf_trt}")
        dim_trt = dim_trt[0, :]
        print(f"TensorRT dim: {dim_trt}")

        # 使用 ONNX 模型進行推論
        input_data = {"input": input_tensor.cpu().numpy()}
        
        output_onnx =  ort_session.run(None, input_data)
        orient_onnx, conf_onnx, dim_onnx = output_onnx

        orient_onnx = orient_onnx[0, :]
        print(f"ONNX orient: {orient_onnx}")
        conf_onnx = conf_onnx[0, :]
        print(f"ONNX conf: {conf_onnx}")
        dim_onnx = dim_onnx[0, :]
        print(f"ONNX dim: {dim_onnx}")

        print(f"Prediction time (TensorRT): {end_time - start_time}")
        time.sleep(1)

if __name__ == "__main__":
    main()