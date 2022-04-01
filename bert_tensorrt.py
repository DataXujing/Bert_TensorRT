import data_processing as dp
import tokenization

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_version = [int(n) for n in trt.__version__.split('.')]

trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# import ctypes


bert_engine = './engines/bert_base_128_zh.engine'
vocab_file = './models/chinese_L-12_H-768_A-12/vocab.txt'
batch_size = 1

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

max_query_length = 64
max_seq_length = 128



max_batch_size = 1

text = '欢迎使用TensorRT!'
input_features = dp.convert_examples_to_features(text, None, tokenizer, max_seq_length)

with open(bert_engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, \
    runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

    print("List engine binding:")
    for binding in engine:
        print(" - {}: {}, Shape {}, {}".format(
            "Input" if engine.binding_is_input(binding) else "Output",
            binding,
            engine.get_binding_shape(binding),
            engine.get_binding_dtype(binding)))

    def binding_nbytes(binding):
        return trt.volume(engine.get_binding_shape(binding)) * engine.get_binding_dtype(binding).itemsize
    
    d_inputs = [cuda.mem_alloc(binding_nbytes(binding)) for binding in engine if engine.binding_is_input(binding)]
    h_output = cuda.pagelocked_empty(tuple(engine.get_binding_shape(3)), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    print("\nRunning Inference...")
    for i in range(10):
        eval_start_time = time.time()
        cuda.memcpy_htod_async(d_inputs[0], input_features["input_ids"], stream)
        cuda.memcpy_htod_async(d_inputs[1], input_features["segment_ids"], stream)
        cuda.memcpy_htod_async(d_inputs[2], input_features["input_mask"], stream)

        context.execute_async(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        eval_time_elapsed = time.time() - eval_start_time

        a = h_output.reshape(128, 768)[0, :]



        print("整体推断耗时： ",eval_time_elapsed * 1000, "ms")
