import torch
import os
import torch.onnx

# 載入模型
model_path = "weights/epoch_10.pkl"
model = torch.load(model_path)

# 指定輸入張量的大小
input_tensor = torch.zeros([1, 3, 224, 224]).cuda()

# 將模型轉換為ONNX格式
torch.onnx.export(model, input_tensor, "epoch_10.onnx", 
                    input_names=["input"], 
                    output_names=["orientation", "confidence", "dimension"],
                    opset_version=11,
                    do_constant_folding=True,
                    export_params=True,
                    keep_initializers_as_inputs=True,
                    dynamic_axes={'input' : {0 : 'batch_size'}, 
                                  'orientation' : {0 : 'batch_size'},
                                  'confidence' : {0 : 'batch_size'},
                                  'dimension' : {0 : 'batch_size'}},
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
