import tensorrt as trt
import numpy as np
import torch
import collections
import cv2
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


out = 'dimension'
w = './epoch_10.engine'
device = torch.device('cuda')
# 1.创建一个Binding对象，该对象包含'name', 'dtype', 'shape', 'data', 'ptr'这些属性
Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
# 2.读取engine文件并记录log
with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    # 将engine进行反序列化，这里的model就是反序列化中的model
    model = runtime.deserialize_cuda_engine(f.read())  # model <class 'tensorrt.tensorrt.ICudaEngine'> num_bindings=2,num_layers=163
# 3.构建可执行的context(上下文：记录执行任务所需要的相关信息)
context = model.create_execution_context()  # <IExecutionContext>
bindings = collections.OrderedDict()
output_names = []
fp16 = True  # default updated below
dynamic = False



for i in range(model.num_bindings):
    name = model.get_binding_name(i) 
    print(name)
    dtype = trt.nptype(model.get_binding_dtype(i))
    if model.binding_is_input(i):  
        if -1 in tuple(model.get_binding_shape(i)):  
            dynamic = True
            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
        if dtype == np.float16:
            fp16 = True
    else:  # output
        output_names.append(name)  # put in the output name
    shape = tuple(context.get_binding_shape(i))  # record the shape of input, output
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # create zero tensor of input shape, output shape
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # put in the Dictionary created before
    #print(bindings)
binding_addrs = collections.OrderedDict((n, d.ptr) for n, d in bindings.items())  #extract names and the bidings

batch_size = bindings[out].shape[0]  # if dynamic, this is instead max batch size
print(f'batch_size:{batch_size}')

#forwarding
s = bindings['input'].shape
print(s)
#assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
#binding_addrs['images'] = int(im.data_ptr())

# 调用计算核心执行计算过程
#context.execute_v2(list(binding_addrs.values()))
#y = [bindings[x].data for x in sorted(output_names)]

#print(y)
'''
#box_2d = y.box_2d

print(y[0])
print(y[0].shape)
#cv2.imshow('3D detections', y[0]) 

'''
'''
if isinstance(y, (list, tuple)):
    from_numpy(y[0]) if len(y) == 1 else [from_numpy(x) for x in y
'''