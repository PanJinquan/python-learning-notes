# -*-coding: utf-8 -*-
"""
    @Project: tf-face-recognition
    @File   : device.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-17 10:39:28
"""
import tensorflow as tf


def set_device_memory(gpu_fraction=0.90):
    tf_config = tf.compat.v1.ConfigProto()
    # tf_config.log_device_placement = False
    # tf_config.allow_soft_placement = allow_soft_placement
    # tf_config.gpu_options.allow_growth = allow_growth
    tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction  # 占用85%显存
    gpu_id = tf.config.experimental.list_physical_devices('GPU')
    if gpu_id:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpu_id:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("use gpu id:{}".format(gpu.name))
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpu_id), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":
    set_device_memory()
