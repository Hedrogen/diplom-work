import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def freeze(model_dir: str, save_dir: str, save_name: str):
    model = tf.saved_model.load(model_dir)

    tfm = tf.function(lambda x: model(x))  # full model
    tfm = tfm.get_concrete_function(
        tf.TensorSpec(
            model.signatures['serving_default'].inputs[0].shape.as_list(),
            model.signatures['serving_default'].inputs[0].dtype.name
        )
    )   
    frozen_func = convert_variables_to_constants_v2(tfm)                                                                                                                              
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=save_dir,
        name=save_name,
        as_text=False
    )