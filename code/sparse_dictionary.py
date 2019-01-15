import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.random as random
import math
import numpy as np


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
        # tf.keras.initializers.O

    # def build(self):
    #     return

    '''thresholding2 implementation'''
    def call(self, input):

        print("<==============================input===========================>")
        print(input)
        print("<==============================tf.shape(input)===========================>")
        print(tf.shape(input))
        print("<==============================input.get_shape()===========================>")
        print(input.get_shape())
        print("<==============================self.dictionary===========================>")
        print(self.dictionary)
        print("<==============================self.l1_coeff===========================>")
        print(self.l1_coeff)
        print("<==============================self.num_embeddings===========================>")
        print(self.num_embeddings)
        print("<==============================self.embeddings_dim===========================>")
        print(self.embeddings_dim)

        batch_size = input.get_shape()[0]
        num_latents = int(np.prod(np.array(input.get_shape()[2:])))
        emb_dim = self.dictionary.get_shape()[0]
        # num_emb = dict.size(1)
        # input_type = type(input)
        print("num_latent = " + str(num_latents))
        #TODO: make sure flatten is executed in the right order
#        x_reshaped = tf.layers.Flatten()(input)
        x_temp = tf.keras.layers.Permute((2,3,1), input_shape=input.get_shape())(input)
        x_reshaped = tf.keras.layers.Reshape([batch_size * num_latents, emb_dim])(x_temp)
        x_reshaped = tf.keras.backend.transpose(x_reshaped)
        # x_reshaped = input.permute(0, 2, 3, 1).contiguous().view(batch_size * num_latents, emb_dim).t()

        print("<==============================x_reshaped.get_shape()==================>")
        print(x_reshaped.get_shape())
        alpha = tf.matmul(self.dictionary, x_reshaped, True)
        # alpha = self.dictionary.t() @ x_reshaped  # kx(nxnxb)
        alpha[math.abs(alpha) < math.sqrt(2*self.l1_coeff)] = 0

        # TODO: add quantization stage
        # alpha = quantize(alpha, num_bits=num_bits, min_value=min_value, max_value=max_value)

        result = self.dictionary @ alpha

        return result.t().contiguous().view(batch_size, *input.size()[2:], emb_dim).permute(0, 3, 1, 2)
