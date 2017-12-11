#-*- coding:utf-8 -*-
from __future__ import print_function
from os.path import dirname
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

"""
1、Analysis model
"""
tf_model_path = '/path/to/tiny-yolo-cel.pb'
with open(tf_model_path, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

# For demonstration purpose we show the first 15 ops the TF model
with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')
    ops = g.get_operations()
    #for i in xrange(15):
    for i in range(len(ops)):
        print('op id {} : op name: {}, op type: "{}"'.format(str(i),ops[i].name, ops[i].type));
print("Analysis Done !")

"""
2、Strip the model
"""
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
input_node_names = ['input']
output_node_names = ['output']
gdef = strip_unused_lib.strip_unused(
        input_graph_def = original_gdef,
        input_node_names = input_node_names,
        output_node_names = output_node_names,
        placeholder_type_enum = dtypes.float32.as_datatype_enum)
# Save it to an output file
frozen_model_file = './frozen-tiny-cel-8250.pb'
with gfile.GFile(frozen_model_file, "wb") as f:
    f.write(gdef.SerializeToString())

print("Strip graph Done !")

"""
3、Convert to mlmodel
"""
import tfcoreml
# Supply a dictionary of input tensors' name and shape (with 
# batch axis)
input_tensor_shapes = {"input:0":[1,416,416,3]} # batch size is 1
# Output CoreML model path
coreml_model_file = './frozen-tiny-cel-8250.mlmodel'
# The TF model's ouput tensor name
output_tensor_names = ['output:0']

# Call the converter. This may take a while
coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = ['input:0'],
        #red_bias = -1,
        #green_bias = -1,
        #blue_bias = -1,
        #image_scale = 2.0/255.0,
        image_scale=1.0/255.0)
print("Convert Done !")

