import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class Length(layers.Layer):
    def get_config(self):
        return super(Length, self).get_config()

    def call(self, inputs, **kwargs):
        return compute_norm(inputs)

    def compute_output_shape(self, input_shape):
        return get_the_input(input_shape)

    

def compute_norm(inputs):
    return K.sqrt(K.sum(K.square(inputs), -1))

def get_the_input(input_shape):
    return input_shape[:-1]

def assert_for_length(inputs):
    assert len(inputs) == 2  

def one_hot_encoding(x):
    return K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])  

def batch_flatten_the_input(inputs, mask):
    return K.batch_flatten(inputs * K.expand_dims(mask, -1))


def return_is_tuple(input_shape):
    return tuple([None, input_shape[0][1] * input_shape[0][2]])

def return_is_not_tuple(input_shape):
    return tuple([None, input_shape[1] * input_shape[2]])

def sum_of_square(vectors, axis):
    return K.sum(K.square(vectors), axis, keepdims=True)

def normalized_the_data(s_squared_norm):
    return s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())

class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert_for_length(inputs)
            inputs, mask = inputs
        else: 
            x = compute_norm(inputs)
            mask = one_hot_encoding(x)

        masked = batch_flatten_the_input(inputs, mask)
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return return_is_tuple(input_shape)
        else:
            return return_is_not_tuple(input_shape)

    def get_config(self):
        return super(Mask, self).get_config()


def get_input(num_capsule, dim_capsule, routings, channels, kernel_initializer):
    return num_capsule, dim_capsule, routings, channels, initializers.get(kernel_initializer)


def squash(vectors, axis=-1):
    s_squared_norm = sum_of_square(vectors, axis)
    scale = normalized_the_data(s_squared_norm)
    return scale * vectors

def get_shape_dim(input_shape):
    return input_shape[1], input_shape[2]

def not_defined_properly(input_shape):
    assert len(input_shape) >= 3, "not defined properly"

def error_in_code(input_num_capsule, channels):
     assert int(input_num_capsule/channels)/(input_num_capsule/channels)==1, "error"


def add_weights_for_layer_1(num_capsule, channels, dim_capsule, input_dim_capsule):
    return [num_capsule, channels, dim_capsule, input_dim_capsule]

def add_bias(num_capsule, dim_capsule):
    return [num_capsule,dim_capsule]  

def add_weights_second_time(num_capsule,input_num_capsule, dim_capsule):
    return [num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]   

def expand_inputs(inputs):
    return K.expand_dims(inputs, 1) 

def input_tile(inputs_expand, num_capsule):
    return K.tile(inputs_expand, [1, num_capsule, 1, 1])


def repeat_elements_value(W, input_num_capsule, channels):
    return K.repeat_elements(W,int(input_num_capsule/channels),1)

def map_the_input(W2, inputs_tiled):
    return K.map_fn(lambda x: K.batch_dot(x, W2, [2, 3]) , elems=inputs_tiled)

def get_the_bias(inputs_hat,num_capsule, input_num_capsule):
    return tf.zeros(shape=[K.shape(inputs_hat)[0], num_capsule, input_num_capsule])

def assert_the_function(routings):
    assert routings > 0, 'invalid input.'

def squash_the_input(inputs_hat, b, B):
    return squash(K.batch_dot(tf.nn.softmax(b, dim=1), inputs_hat, [2, 2])+ B)      

def create_tuple(arr):
    return tuple(arr)     

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule,channels, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule, self.dim_capsule, self.routings, self.channels, self.kernel_initializer = get_input(num_capsule, dim_capsule, routings, channels, kernel_initializer)

    def build(self, input_shape):
        not_defined_properly(input_shape)
        self.input_num_capsule, self.input_dim_capsule =  get_shape_dim(input_shape)
        
        if(self.channels!=0):
            error_in_code(self.input_num_capsule, self.channels)
            self.W = self.add_weight(shape=add_weights_for_layer_1(self.num_capsule, self.channels, self.dim_capsule, self.input_dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='W')
            
            self.B = self.add_weight(shape=add_bias(self.num_capsule, self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='B')
        else:
            self.W = self.add_weight(shape=add_weights_second_time(self.num_capsule,self.input_num_capsule, self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='W')
            self.B = self.add_weight(shape=add_bias(self.num_capsule, self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='B')
        self.built = True

      

    def call(self, inputs, training=None):
        inputs_expand = expand_inputs(inputs)
        
        inputs_tiled = input_tile(inputs_expand, self.num_capsule)
        
        if(self.channels!=0):
            W2 = repeat_elements_value(self.W, self.input_num_capsule, self.channels)
        else:
            W2 = self.W
            
        inputs_hat = map_the_input(W2, inputs_tiled)

        b = get_the_bias(inputs_hat,self.num_capsule, self.input_num_capsule)

        assert_the_function(self.routings)
        for i in range(self.routings):
            outputs = squash_the_input(inputs_hat, b, self.B)

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return create_tuple([None, self.num_capsule, self.dim_capsule])

def calculate_filter_value(dim_capsule,n_channels):
    return dim_capsule*n_channels

def target_shape_get_it(dim_capsule):
    return [-1, dim_capsule]

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=calculate_filter_value(dim_capsule,n_channels), kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=target_shape_get_it(dim_capsule), name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)