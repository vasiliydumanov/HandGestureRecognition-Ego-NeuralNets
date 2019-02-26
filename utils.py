
from tensorflow.python.tools.freeze_graph import freeze_graph
import coremltools.proto.FeatureTypes_pb2 as ft


def input_output_to_float32(spec):
    for feature in spec.description.output:
        update_multiarray_to_float32(feature)
    for feature in spec.description.input:
        update_multiarray_to_float32(feature)


def update_multiarray_to_float32(feature):
    if feature.type.HasField("multiArrayType"):
        feature.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32


def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")


def rename_input(spec, input, new_name):
    old_name = input.name
    input.name = new_name
    nn = get_nn(spec)
    for i in range(len(nn.layers)):
        for k in range(len(nn.layers[i].input)):
            if nn.layers[i].input[k] == old_name:
                nn.layers[i].input[k] = new_name


def rename_output(spec, output, new_name):
    old_name = output.name
    output.name = new_name
    nn = get_nn(spec)
    for i in range(len(nn.layers)):
        for k in range(len(nn.layers[i].output)):
            if nn.layers[i].output[k] == old_name:
                nn.layers[i].output[k] = new_name


