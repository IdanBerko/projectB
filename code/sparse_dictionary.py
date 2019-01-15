import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.random as random
import math
import numpy as np


class SparseDictionary(keras.layers.Layer):

    def __init__(self, num_embeddings, embeddings_dim, l1_coeff=0):
        super(SparseDictionary, self, trainable=False, name="Sparse Dictionary", dtype=None).__init__()

        self.l1_coeff = l1_coeff
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim

        # self.dictionary = self.add_variable("Dictionary Values", shape=[embeddings_dim, num_embeddings])

        random_dictionary = random.uniform(shape=[embeddings_dim, num_embeddings])
        self.dictionary = tf.Variable(tf.math.l2_normalize(random_dictionary))
        keras.initializers.orthogonal(self.dictionary)

    def build(self):
        return

    '''thresholding2 implementation'''
    def call(self, input):
        print("<==============================input.size===========================>")
        print(input.size())
        batch_size = input.size(0)
        num_latents = int(np.prod(np.array(input.size()[2:])))
        emb_dim = self.dictionary.size(0)
        # num_emb = dict.size(1)
        # input_type = type(input)

        x_reshaped = input.permute(0, 2, 3, 1).contiguous().view(batch_size * num_latents, emb_dim).t()
        alpha = self.dictionary.t() @ x_reshaped  # kx(nxnxb)
        alpha[math.abs(alpha) < math.sqrt(2*self.l1_coeff)] = 0

        # TODO: add quantization stage
        # alpha = quantize(alpha, num_bits=num_bits, min_value=min_value, max_value=max_value)

        result = self.dictionary @ alpha

        return result.t().contiguous().view(batch_size, *input.size()[2:], emb_dim).permute(0, 3, 1, 2)
