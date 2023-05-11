import os
from rknn.api import RKNN
import numpy as np
import onnxruntime as ort

onnx_model = 'static_det.onnx' #onnx路径
save_rknn_dir = 'static_det.rknn'#rknn保存路径

def norm(img):
    mean = 0.5
    std = 0.5
    img_data = (img.astype(np.float32)/255 - mean) / std
    return img_data

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # image = np.random.randn(1,3,32,448).astype(np.float32)
    image = np.random.randn(1,3,640,640).astype(np.float32)        # 创建一个np数组，分别用onnx和rknn推理看转换后的输出差异，检测模型输入是1，3，640，640 ，识别模型输入是1，3，32，448
    onnx_net = ort.InferenceSession(onnx_model)                   # onnx推理
    onnx_infer = onnx_net.run(None, {'x': norm(image)})       # 如果是paddle2onnx转出来的模型输入名字默认是 "x"
    # onnx_infer = onnx_net.run(None, {'input': norm(image)})             

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], reorder_channel='2 1 0', target_platform=['rv1126'], batch_size=4,quantized_dtype='asymmetric_quantized-u8')  # 需要输入为RGB#####需要转化一下均值和归一化的值
    # rknn.config(mean_values=[[0.0, 0.0, 0.0]], std_values=[[255, 255, 255]], reorder_channel='2 1 0', target_platform=['rv1126'], batch_size=1)  # 需要输入为RGB
    print('done')

    # model_name = onnx_model[onnx_model.rfind('/') + 1:]
    # Load ONNX model
    print('--> Loading model %s' % onnx_model)
    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        print('Load %s failed!' % onnx_model)
        exit(ret)
    print('done')
    # Build model
    print('--> Building model')
    # rknn.build(do_quantization=False)
    ret = rknn.build(do_quantization=True, dataset='dataset.txt', pre_compile=False)
    #do_quantization是否对模型进行量化，datase量化校正数据集，pre_compil模型预编译开关，预编译 RKNN 模型可以减少模型初始化时间，但是无法通过模拟器进行推理或性能评估
    if ret != 0:
        print('Build net failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(save_rknn_dir)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)

    ret = rknn.init_runtime(target='rv1126',device_id="a9d00ab1f032c17a")            # 两个参数分别是板子型号和device_id,device_id在双头usb线连接后通过 adb devices查看
    if ret != 0:
        print('init runtime failed.')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[image])

    # perf
    print('--> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[image])              # 模型评估
    print('done')
    print()

    print("->>模型前向对比！")
    print("--------------rknn outputs--------------------")
    print(outputs[0])
    print()

    print("--------------onnx outputs--------------------")
    print(onnx_infer[0])
    print()

    std = np.std(outputs[0]-onnx_infer[0])
    print(std)                                # 如果这个值比较大的话，说明模型转换后不太理想

    rknn.release()

