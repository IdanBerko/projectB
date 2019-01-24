import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.random as random
import math
import numpy as np
from keras.backend import manual_variable_initialization
# import sys

# added recenetly
manual_variable_initialization(True)

class SparseDictionary(tf.keras.layers.Layer):

    def __init__(self, num_embeddings, embeddings_dim, l1_coeff=0):
        super(SparseDictionary, self).__init__()
        self.trainable = False
        # self.name = "Sparse_Dictionary"
        # self.dtype = None

        self.l1_coeff = l1_coeff
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim

        # self.dictionary = self.add_variable("Dictionary Values", shape=[embeddings_dim, num_embeddings])

        random_dictionary = tf.random_uniform(shape=[embeddings_dim, num_embeddings])
        self.dictionary = tf.Variable(tf.nn.l2_normalize(random_dictionary, dim=0))
        tf.keras.initializers.Orthogonal(self.dictionary)
        # normalized_dict = tf.nn.l2_normalize(random_dictionary, dim=0)
        # orthogonal_dict = tf.keras.initializers.Orthogonal(normalized_dict)
        #TODO consider adding dtype = float
        # self.add_weight("dictionary", shape=[embeddings_dim, num_embeddings], initializer=orthogonal_dict)
       # self.dictionary = orthogonal_dict
     #   print(self.get_weights()[0])

        #print(self.dictionary)

    # def build(self):
    #     return

    '''thresholding2 implementation'''
    def call(self, input):
        # print(self.get_weights())
        # dictionary = self.get_weights()[0]
        batch_size = input.get_shape()[0]
        num_latents = int(np.prod(np.array(input.get_shape()[2:])))
        emb_dim = self.dictionary.get_shape()[0]
        #emb_dim = dictionary.shape[0]
        #print(emb_dim)
        #TODO: make sure flatten is executed in the right order

        x_temp = tf.keras.layers.Permute((2,3,1), input_shape=input.get_shape())(input)
        x_reshaped = tf.keras.layers.Reshape([batch_size * num_latents, emb_dim])(x_temp)
        x_reshaped = tf.squeeze(tf.keras.backend.transpose(x_reshaped))

        alpha_temp = tf.matmul(self.dictionary, x_reshaped, True) # True here means transpose dictionary
        thresh = math.sqrt(2*self.l1_coeff)*np.ones(np.shape(alpha_temp))
        cond = tf.less(np.abs(alpha_temp), thresh)
        alpha = tf.where(cond, tf.zeros(tf.shape(alpha_temp)), alpha_temp)

        # TODO: add quantization stage
        # alpha = quantize(alpha, num_bits=num_bits, min_value=min_value, max_value=max_value)
        result = tf.matmul(self.dictionary, alpha)
        res_out_temp = tf.keras.layers.Reshape([batch_size, *input.get_shape()[2:], emb_dim])(tf.transpose(result))
        res_out = tf.squeeze(tf.keras.layers.Permute((1, 4, 2, 3))(res_out_temp))


        #init = tf.global_variables_initializer()

        #with tf.Session() as sess:
        #    sess.run(init)
        #    v = sess.run(self.dictionary)
        #    print(v)  # will show you your variable.


        return res_out
