# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: TF-Demo
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-03-03 10:57:00
# --------------------------------------------------------
"""

import os
import tensorflow as tf
import tensorflow.lite as lite
from tensorflow.python.framework import graph_util

print("TF:{}".format(tf.__version__))


def convert_tflite(pb_model, input_shape, out_tflite=None, quantize=True, optimize=True):
    """
    :param pb_model: TF的*.PB模型路径(固化模型)
    :param input_shape: <list> 输入维度,[112,112]
    :param out_tflite: <str>,default None,输出tflite模型路径
    :param quantize: <bool> 是否进行模型量化
    :param optimize: <bool> 是否进行模型优化
    :return:
    """
    input_arrays = ['Placeholder'] # 定义输入的节点名称
    output_arrays = ['out/BiasAdd'] # 定义输出的节点名称
    input_shapes = {"Placeholder": (1, input_shape[0], input_shape[1], 3)}
    ############################################################################
    # Converting a GraphDef from session.
    # converter = lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)
    ############################################################################
    # Converting a GraphDef from file.
    TFLiteConverter = tf.compat.v1.lite.TFLiteConverter  # for TF2.X
    # TFLiteConverter=tf.lite.TFLiteConverter            # for TF1.X
    converter = TFLiteConverter.from_frozen_graph(graph_def_file=pb_model,
                                                  input_arrays=input_arrays,
                                                  output_arrays=output_arrays,
                                                  input_shapes=input_shapes)
    ############################################################################
    # Converting a SavedModel.
    # converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # tflite_model = converter.convert()
    ############################################################################
    # Converting a tf.keras model.
    # converter = lite.TFLiteConverter.from_keras_model_file(keras_model)
    # tflite_model = converter.convert()
    ############################################################################
    if not out_tflite:
        out_dir = os.path.dirname(pb_model)
        basename = os.path.basename(pb_model).split(".")[:-1]
    else:
        out_dir = os.path.dirname(out_tflite)
        basename = os.path.basename(out_tflite).split(".")[:-1]
    if quantize: # 是否进行量化
        converter.post_training_quantize = True
        basename += ["quantize"]
    if optimize: # 是否进行优化
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        basename += ["optimize"]
    #
    basename = "_".join(basename)
    out_tflite = os.path.join(out_dir, '{}_h{}_w{}.tflite'.format(basename, input_shape[0], input_shape[1]))
    tflite_model = converter.convert()
    open(out_tflite, "wb").write(tflite_model)
    print("save tflite: {}".format(out_tflite))


if __name__ == "__main__":
    # input_shape = [256, 192]
    input_shape = [192, 144]
    # pb_model = "./output/pretrained/COCO/ResNet18/model.pb"
    pb_model = "./work_dir/mobilenet_v2_1.0_64_64_64/model_dump/COCO/model.pb"
    # pb_model = "./work_dir/mobilenet_v2/model_dump/COCO/model.pb"
    convert_tflite(pb_model, input_shape=input_shape, out_tflite=None)
