import tensorflow as tf
import tensorlayer as tl
import numpy as np

def HDRcnn(x):

    inputArray = tf.scalar_mul(255.0,x)
    cnnInput = tl.layers.InputLayer(inputArray, name="Layer_1")
    #print net_in.outputs (debug)
    conv_output, skip_connections = encode_cnn(cnnInput)

    # Fully convolutional layers on top of VGG16 conv layers
    cnn = tl.layers.Conv2dLayer(conv_output, skip_connections,
                    act = tf.identity,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='encoder/h6/conv')

    cnn = tl.layers.BatchNormLayer(cnn, is_train=False, name='Batch_Norm_Layer')
    cnn.outputs = tf.nn.relu(cnn.outputs, name='Relu_Activation')

    cnn = decode_cnn(cnn, skip_connections)

    return cnn

# Get the predictions from the model
def get_prediction(cnn, inputArr):
    sb, sy, sx, sf = inputArr.get_shape().as_list()
    predict = cnn.outputs

    #this creates a highlight mask for our image for alph blending.
    offset= 0.05
    alph = tf.reduce_max(inputArr, reduction_indices=[3])
    alph = tf.minimum(1.0, tf.maximum(0.0, alph-1.0+offset)/offset)
    alph = tf.reshape(alph, [-1, sy, sx, 1])
    alph = tf.tile(alph, [1, 1, 1, 3])

    # We want to "linearize" our prediction and input
    x_linear = tf.pow(inputArr, 2.0)
    predict = tf.exp(predict)-1.0/255.0

    # This does something called "alph Blending" for clearer output.
    # Prevents "banding artifacts" between highlighted areas and their surroundings
    predict = (1-alph)*x_linear + alph*predict
    
    return predict


# Convolutional layers of the VGG16 model used as encoder cnn
def encode_cnn(cnn_Input):
    r, g, b = tf.split(cnn_Input.outputs, 3, 3)
    #offset to mean of VGG BGR Values
    bgr = tf.concat([b - 103.939, g - 116.779, r - 123.68], axis=3)

    cnn = tl.layers.InputLayer(bgr, name='BGR_Encode')

    cnn = add_convolution(cnn, [ 3, 64], 'add_convolution1')
    skip1 = add_convolution(cnn, [64, 64], 'Conv_Layer2')
    cnn = add_pooling(skip1, 'Pool_Layer1')

    cnn = add_convolution(cnn, [64, 128], 'Conv_Layer3')
    skip2 = add_convolution(cnn, [128, 128], 'Conv_Layer4')
    cnn = add_pooling(skip2, 'Pool_Layer2')

    cnn = add_convolution(cnn, [128, 256], 'Conv_Layer5')
    cnn = add_convolution(cnn, [256, 256], 'Conv_Layer6')
    skip3 = add_convolution(cnn, [256, 256], 'Conv_Layer7')
    cnn = add_pooling(skip3, 'Pool_Layer3')

    cnn = add_convolution(cnn, [256, 512], 'Conv_Layer8')
    cnn = add_convolution(cnn, [512, 512], 'Conv_Layer9')
    skip4 = add_convolution(cnn, [512, 512], 'Conv_Layer10')
    cnn = add_pooling(skip4, 'Pool_Layer4')

    cnn = add_convolution(cnn, [512, 512], 'Conv_Layer11')
    cnn = add_convolution(cnn, [512, 512], 'Conv_Layer12')
    skip5 = add_convolution(cnn, [512, 512], 'Conv_Layer13')
    cnn = add_pooling(skip5, 'Pool_Layer5')

    return cnn, (cnn_Input, skip1, skip2, skip3, skip4, skip5)


# Decoder, set to train.
def decode_cnn(cnn_input, skip_connections, batch_size=1, train=True):
    sb, sx, sy, sf = cnn_input.outputs.get_shape().as_list()
    alph = 0.0

    cnn = add_deconv(cnn_input, (batch_size,sx,sy,sf,sf), 'Deconv_Layer1', alph, train)

    cnn = add_skip(cnn, skip_connections[5], 'Skip_1', train)
    cnn = add_deconv(cnn, (batch_size,2*sx,2*sy,sf,sf), 'Deconv_layer2', alph, train)

    cnn = add_skip(cnn, skip_connections[4], 'Skip_2', train)
    cnn = add_deconv(cnn, (batch_size,4*sx,4*sy,sf,sf/2), 'Deconv_Layer3', alph, train)

    cnn = add_skip(cnn, skip_connections[3], 'Skip_3', train)
    cnn = add_deconv(cnn, (batch_size,8*sx,8*sy,sf/2,sf/4), 'Deconv_Layer4', alph, train)

    cnn = add_skip(cnn, skip_connections[2], 'Skip_4', train)
    cnn = add_deconv(cnn, (batch_size,16*sx,16*sy,sf/4,sf/8), 'Deconv_Layer5', alph, train)

    cnn = add_skip(cnn, skip_connections[1], 'Skip_5', train)

    cnn = tl.layers.Conv2dLayer(cnn,
                        act = tf.identity,
                        shape = [1, 1, int(sf/8), 3],
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init = tf.contrib.layers.xavier_initializer(uniform=False),
                        b_init = tf.constant_initializer(value=0.0),
                        name ='Last Conv')

    cnn = tl.layers.BatchNormLayer(cnn, is_train=train, name='Batch_Norm_layer')
    cnn.outputs = tf.maximum(alph*cnn.outputs, cnn.outputs, name='Leaky_Relu_Fix')
    cnn = add_skip(cnn, skip_connections[0], 'Skip_6')
    return cnn


def add_convolution(cnn_input, size, str):
    cnn = tl.layers.Conv2dLayer(cnn_input,
                    act = tf.nn.relu,
                    shape = [3, 3, size[0], size[1]],
                    strides = [1, 1, 1, 1],
                    padding = 'SAME',
                    name = str)

    return cnn


def add_pooling(cnn_input, str):
    cnn = tl.layers.PoolLayer(cnn_input,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name = str)

    return cnn



def add_skip(cnn_input, skip_layer, str, train=True):
    batch, sx, sy, sf = cnn_input.outputs.get_shape().as_list()
    batch, sx_, sy_, sf_ = skip_layer.outputs.get_shape().as_list()
    #make sure the sizes are equal for the skip connection layers and input layer
    assert (sx_,sy_,sf_) == (sx,sy,sf)

    skip_layer.outputs = tf.log(tf.pow(tf.scalar_mul(1.0/255, skip_layer.outputs), 2.0)+1.0/255.0)

    cnn = tl.layers.ConcatLayer(layer = [cnn_input,skip_layer], concat_dim=3, name ='%s/skip_connection'%str)

    cnn = tl.layers.Conv2dLayer(cnn,
                    act = tf.identity,
                    shape = [1, 1, sf+sf_, sf],
                    strides = [1, 1, 1, 1],
                    padding = 'SAME',
                    b_init = tf.constant_initializer(value=0.0),
                    name = str)

    return cnn


def add_deconv(cnn_input, size, str, alph, train=True):
    scale = 2

    filter_size = (2 * scale - scale % 2)
    in_channels = int(size[3])
    out_channels = int(size[4])

    cnn = tl.layers.DeConv2dLayer(cnn_input,
                                shape = [filter_size, filter_size, out_channels, in_channels],
                                output_shape = [size[0], size[1]*scale, size[2]*scale, out_channels],
                                strides = [1, scale, scale, 1],
                                padding = 'SAME',
                                act = tf.identity,
                                name = str)

    cnn = tl.layers.BatchNormLayer(cnn, is_train=train, name='Batch_Norm_Layer')
    cnn.outputs = tf.maximum(alph*cnn.outputs, cnn.outputs, name='Leaky_Relu_Layer')

    return cnn

