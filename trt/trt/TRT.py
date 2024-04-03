import tensorrt as trt
import numpy as np
import torch
import collections
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
CUDA_LAUNCH_BLOCKING=1
class TRT():
    def __init__(self, model_path = None, fp16 = False) -> None:
        super(TRT, self).__init__()
        self.model_path = model_path
        self.device = torch.device('cuda')
        self.Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        self.fp16 = fp16
        self.binding_addrs = None
        self.model = None
        self.bindings = collections.OrderedDict()
        self.output_names = []
        self.dynamic = False
        self.name = None
        self.batch_size = 1
        self.im = None
        self.context = None
    def start(self)-> None:
        with open(self.model_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()

        for i in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(i) 
            #print(name)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            if self.model.binding_is_input(i):  
                if -1 in tuple(self.model.get_binding_shape(i)):  
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            else:  # output
                self.output_names.append(self.name)  # put in the output name
            shape = tuple(self.context.get_binding_shape(i))  # record the shape of input, output
            self.im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)  # create zero tensor of input shape, output shape
            self.bindings[self.name] = self.Binding(self.name, dtype, shape, self.im, int(self.im.data_ptr()))  # put in the Dictionary created before
            #print(bindings)
        self.binding_addrs = collections.OrderedDict((n, d.ptr) for n, d in self.bindings.items())  #extract names and the bidings
    def predict(self, input_tensor):
        input_name = self.model.get_binding_name(0) 
        input_shape = self.context.get_binding_shape(0)
        assert input_tensor.shape == tuple(input_shape), f"input size {input_tensor.shape} not equal to model input size {input_shape}"
        
        input_np = input_tensor.cpu().numpy().astype(trt.nptype(self.model.get_binding_dtype(0)))
        
        self.binding_addrs['input'] = int(self.im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        output = [self.bindings[x].data for x in sorted(self.output_names)]
        print(output)
        output= [tensor.cpu().numpy() for tensor in output]
        return output