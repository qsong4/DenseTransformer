import tensorflow as tf

from data_load import load_vocab, loadGloVe, load_char_vocab
from modules import get_token_embeddings, ff, positional_encoding, CNN_3d, CNN, multihead_attention, inter_multihead_attention, ln, positional_encoding_bert, noam_scheme
from tensorflow.python.ops import nn_ops


class FI:
    """
    xs: tuple of
        x: int32 tensor. (句子长度，)
        x_seqlens. int32 tensor. (句子)
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token, self.hp.vocab_size = load_vocab(hp.vocab)
        self.embd = None
        if self.hp.preembedding:
            self.embd = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embd, self.hp.vocab_size, self.hp.d_model, zero_pad=False)
        self.x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])
        self.y_len = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None, self.hp.num_class], name="truth")
        self.is_training = tf.placeholder(tf.bool,shape=None, name="is_training")

        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_op()
        self.train = self._training_op()

    def create_feed_dict(self, x, y, x_len, y_len, truth, is_training):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
            self.truth: truth,
            self.is_training: is_training,
        }

        return feed_dict


    def create_feed_dict_infer(self, x, y, x_len, y_len):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
        }

        return feed_dict


    def representation(self, xs, ys):
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            x = xs
            y = ys

            # print(x)
            # print(y)


            # embedding
            encx = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            encx *= self.hp.d_model ** 0.5  # scale

            encx += positional_encoding_bert(encx, self.hp.maxlen)
            encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=self.is_training)

            ency = tf.nn.embedding_lookup(self.embeddings, y)  # (N, T1, d_model)
            ency *= self.hp.d_model ** 0.5  # scale

            ency += positional_encoding_bert(ency, self.hp.maxlen)
            ency = tf.layers.dropout(ency, self.hp.dropout_rate, training=self.is_training)
            # add ln
            encx = ln(encx)
            ency = ln(ency)

            x_layer = []
            y_layer = []

            # 这两个模块可以互换
            # Inter Inference Block
            #for i in range(self.hp.num_inter_blocks):
                #encx, ency = self.inter_blocks(encx, ency, x_layer, y_layer, i, scope="num_inter_blocks_{}".format(i))
                #x_layer.append(encx)
                #y_layer.append(ency)


            # Inference Block
            for i in range(self.hp.inference_blocks):
                encx, ency = self.dense_blocks(encx, ency, scope="num_inference_blocks_{}".format(i))
                x_layer.append(encx)
                y_layer.append(ency)

        print("x_layer: ", x_layer[0].shape)
        print(len(x_layer))
        return x_layer, y_layer
        #return encx, encx

    def dense_blocks(self, a_repre, b_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            _encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention
            _ency = multihead_attention(queries=b_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)



            # inter-attention
            ency = multihead_attention(queries=_encx,
                                       keys=_ency,
                                       values=_ency,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # inter-attention
            encx = multihead_attention(queries=_ency,
                                       keys=_encx,
                                       values=_encx,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            encx, ency = self._infer(encx, ency)

            #encx, ency, ae_loss = self._dense_infer(encx, ency, x_layer, y_layer, layer_num)

            return encx, ency

    def inference_blocks(self, a_repre, b_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention
            ency = multihead_attention(queries=b_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            encx, ency = self._infer(encx, ency)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
            ency = ff(ency, num_units=[self.hp.d_ff, self.hp.d_model])

            #先进行infer然后再过全连接
            #encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
            #ency = ff(ency, num_units=[self.hp.d_ff, self.hp.d_model])

            return encx, ency

    def calcuate_attention(self, in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                           att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None):
        input_shape = tf.shape(in_value_1)
        batch_size = input_shape[0]
        len_1 = input_shape[1]
        len_2 = tf.shape(in_value_2)[1]

        #in_value_1 = tf.layers.dropout(in_value_1, hp.dropout_rate, training=training)
        #in_value_2 = tf.layers.dropout(in_value_2, hp.dropout_rate, training=training)
        with tf.variable_scope(scope_name):
            # calculate attention ==> a: [batch_size, len_1, len_2]
            atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
            if feature_dim1 == feature_dim2:
                atten_w2 = atten_w1
            else:
                atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
            atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]),
                                      atten_w1)  # [batch_size*len_1, feature_dim]
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]),
                                      atten_w2)  # [batch_size*len_2, feature_dim]
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

            if att_type == 'additive':
                atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
                atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
                atten_value_1 = tf.expand_dims(atten_value_1, axis=2,
                                               name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
                atten_value_2 = tf.expand_dims(atten_value_2, axis=1,
                                               name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
                atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
                atten_value = nn_ops.bias_add(atten_value, atten_b)
                atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
                atten_value = tf.reshape(atten_value, [-1,
                                                       att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
                atten_value = tf.reduce_sum(atten_value, axis=-1)
                atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
            else:
                atten_value_1 = tf.tanh(atten_value_1)
                # atten_value_1 = tf.nn.relu(atten_value_1)
                atten_value_2 = tf.tanh(atten_value_2)
                # atten_value_2 = tf.nn.relu(atten_value_2)
                diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
                atten_value_1 = atten_value_1 * diagnoal_params
                atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True)  # [batch_size, len_1, len_2]

            # normalize
            if remove_diagnoal:
                diagnoal = tf.ones([len_1], tf.float32)  # [len1]
                diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
                diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
                atten_value = atten_value * diagnoal
            if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
            if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
            atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
            if remove_diagnoal: atten_value = atten_value * diagnoal
            if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
            if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

        return atten_value

    def cross_attention(self, a_repre, b_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            encx = multihead_attention(queries=b_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, encx.shape.as_list()[-1]])

            # self-attention
            ency = multihead_attention(queries=a_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            encx, ency = self._infer(encx, ency)

            # feed forward
            ency = ff(ency, num_units=[self.hp.d_ff, encx.shape.as_list()[-1]])

        return encx, ency

    def calculate_att(self, encx, ency, scope):
        with tf.variable_scope(scope):
            x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
            y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)
            # match_result_x = match_passage_with_question(encx, ency, x_mask, y_mask)
            # match_result_y = match_passage_with_question(ency, encx, x_mask, y_mask)

            attentionWeights = self.calcuate_attention(encx, ency, encx.shape.as_list()[-1], ency.shape.as_list()[-1],
                                              scope_name="attention", att_type=self.hp.att_type, att_dim=self.hp.att_dim,
                                              remove_diagnoal=False, mask1=x_mask, mask2=y_mask)
            #attentionWeights = tf.matmul(encx, tf.transpose(ency, [0, 2, 1]))
            attentionSoft_a = tf.nn.softmax(attentionWeights)
            attentionSoft_b = tf.nn.softmax(tf.transpose(attentionWeights))
            attentionSoft_b = tf.transpose(attentionSoft_b)
            #print(attentionSoft_a.shape)
            #a_hat = tf.matmul(attentionSoft_a, ency)
            #b_hat = tf.matmul(attentionSoft_b, encx)
        return attentionSoft_a, attentionSoft_b

    def _dense_infer(self, encx, ency, x_layer, y_layer, layer_num, scope="dese_local_inference"):
        with tf.variable_scope(scope):
            #x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
            #y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)

            #cross_encx, cross_ency = self.calculate_att(encx, ency, scope="cal_att")
            #_encx, _ency = self._infer(encx, ency)
            #print(cross_encx.shape)
            #print(cross_ency.shape)
            #可以有两种方式
            #1. concat前面所有层的信息
            #2. 只concat前面一层的信息
            a_res = tf.concat([x_layer[-1]] + [encx], axis=2)
            b_res = tf.concat([y_layer[-1]] + [ency], axis=2)
            #a_res = tf.layers.dropout(a_res, self.hp.dropout_rate, training=self.is_training)
            #b_res = tf.layers.dropout(b_res, self.hp.dropout_rate, training=self.is_training)
            #if layer_num in self.AE_layer:
            a_res = self._project_op(a_res)  # (?,?,d_model)
            b_res = self._project_op(b_res)  # (?,?,d_model)
            ae_loss_a = 0
            ae_loss_b = 0
            # if layer_num in self.AE_layer:
            #     a_res, ae_loss_a = self._AutoEncoder(a_res)
            #     b_res, ae_loss_b = self._AutoEncoder(b_res)

        return a_res, b_res, ae_loss_a + ae_loss_b


    def _infer(self, encx, ency, scope="local_inference"):
        with tf.variable_scope(scope):
            # x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
            # y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)
            # match_result_x = match_passage_with_question(encx, ency, x_mask, y_mask)
            # match_result_y = match_passage_with_question(ency, encx, x_mask, y_mask)

            # attentionWeights = self.calcuate_attention(encx, ency, self.hp.d_model, self.hp.d_model,
            #                                   scope_name="attention", att_type=self.hp.att_type, att_dim=self.hp.att_dim,
            #                                   remove_diagnoal=False, mask1=x_mask, mask2=y_mask)

            attentionWeights = tf.matmul(encx, tf.transpose(ency, [0, 2, 1]))
            attentionSoft_a = tf.nn.softmax(attentionWeights)
            attentionSoft_b = tf.nn.softmax(tf.transpose(attentionWeights))
            attentionSoft_b = tf.transpose(attentionSoft_b)

            a_hat = tf.matmul(attentionSoft_a, ency)
            b_hat = tf.matmul(attentionSoft_b, encx)
            a_diff = tf.subtract(encx, a_hat)
            a_mul = tf.multiply(encx, a_hat)
            b_diff = tf.subtract(ency, b_hat)
            b_mul = tf.multiply(ency, b_hat)

            a_res = tf.concat([a_hat, a_diff, a_mul], axis=2)
            b_res = tf.concat([b_hat, b_diff, b_mul], axis=2)

            # BN
            # a_res = tf.layers.batch_normalization(a_res, training=self.is_training, name='bn1', reuse=tf.AUTO_REUSE)
            # b_res = tf.layers.batch_normalization(b_res, training=self.is_training, name='bn2', reuse=tf.AUTO_REUSE)
            # project
            a_res = self._project_op(a_res)  # (?,?,d_model)
            b_res = self._project_op(b_res)  # (?,?,d_model)

            a_res += encx
            b_res += ency
            #a_res = tf.concat([encx, a_res], axis=-1)
            #b_res = tf.concat([ency, b_res], axis=-1)

            a_res = ln(a_res)
            b_res = ln(b_res)

        return a_res, b_res

    def fc_2l(self, inputs, num_units, scope="fc_2l"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

        return outputs

    def _project_op(self, inputx):
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            inputx = tf.layers.dense(inputx, self.hp.d_model,
                                     activation=tf.nn.relu,
                                     name='fnn',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            return inputx

    def _AutoEncoder(self, inputx):
        with tf.variable_scope("AE", reuse=tf.AUTO_REUSE):
            # # Network variables
            # encoder_weights = tf.Variable(tf.random_normal(shape=(self.features.shape[1], n_dimensions)))
            # encoder_bias = tf.Variable(tf.zeros(shape=[n_dimensions]))
            #
            # decoder_weights = tf.Variable(tf.random_normal(shape=(n_dimensions, self.features.shape[1])))
            # decoder_bias = tf.Variable(tf.zeros(shape=[self.features.shape[1]]))
            #
            # # Encoder part
            # encoding = tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_weights), encoder_bias))
            #
            # # Decoder part
            # predicted_x = tf.nn.sigmoid(tf.add(tf.matmul(encoding, decoder_weights), decoder_bias))
            dim = inputx.shape.as_list()[-1]
            encoder_result = tf.layers.dense(inputx, self.hp.d_model,
                                     activation=tf.nn.relu,
                                     name='ae_encoder',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            decoder_result = tf.layers.dense(inputx, dim,
                                     activation=tf.nn.relu,
                                     name='ae_decoder',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            # Define the cost function and optimizer to minimize squared error
            ae_loss = tf.reduce_mean(tf.pow(tf.subtract(decoder_result, inputx), 2))

            return encoder_result, ae_loss
            #optimizer = tf.train.AdamOptimizer().minimize(cost)


    def base_blocks(self, a_repre, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # self-attention
            encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
        return encx

    def inter_blocks(self, a_repre, b_repre, x_layer, y_layer, layer_num, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # encx, ency = inter_multihead_attention(queries=b_repre,
            #                            keys=a_repre,
            #                            values=a_repre,
            #                            num_heads=self.hp.num_heads,
            #                            dropout_rate=self.hp.dropout_rate,
            #                            training=self.is_training,
            #                            causality=False)

            # self-attention
            encx = multihead_attention(queries=b_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            #feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])

            # self-attention
            ency = multihead_attention(queries=a_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            #feed forward
            ency = ff(ency, num_units=[self.hp.d_ff, self.hp.d_model])

            #encx, ency, ae_loss = self._dense_infer(encx, ency, x_layer, y_layer, layer_num)

            #encx, ency = self._infer(encx, ency)

        return encx, ency

    def aggregation(self, a_repre, b_repre):
        dim = a_repre.shape.as_list()[-1]
        with tf.variable_scope("aggregation", reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.hp.num_agg_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Vanilla attention
                    a_repre = multihead_attention(queries=a_repre,
                                                  keys=a_repre,
                                                  values=a_repre,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=self.is_training,
                                                  causality=False,
                                                  scope="vanilla_attention")
                    ### Feed Forward
                    a_repre = ff(a_repre, num_units=[self.hp.d_ff, dim])
        return a_repre

    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b
        # prob = tf.nn.softmax(logits)

        # gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return logits

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # 将transformer的每层作为一个channel输入CNN
    def cnn_agg(self, match_channels):
        # Create a convolution + maxpool layer for each filter size
        filter_sizes = list(map(int, self.hp.filter_sizes.split(",")))
        embedding_size = match_channels.shape.as_list()[2]
        sequence_length = match_channels.shape.as_list()[1]
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 6, self.hp.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.hp.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    match_channels,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.hp.num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.hp.dropout_rate)

        return h_drop

    def _global_avg_pooling(self, net):
        net = tf.reduce_mean(net, [1, 2], name='gap', keep_dims=True)
        return net

    def _logits_op(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y)  # (layers, batchsize, maxlen, d_model)

        x_repre = tf.stack(x_repre, axis=-1)
        y_repre = tf.stack(y_repre, axis=-1)

        print("x_repre shape:", x_repre.shape)
        print("y_repre shape:", y_repre.shape)

        concat_xy = tf.concat([x_repre, y_repre], axis=-1)

        print("concat_xy shape:", concat_xy.shape)

        #concat_xy = self._project_op(concat_xy)
        _gap = self._global_avg_pooling(concat_xy)
        gap_res = tf.reshape(_gap,[-1,_gap.shape.as_list()[-1]])
        print("gap_res shape:", gap_res.shape)

        logits = self.fc(gap_res, match_dim=gap_res.shape.as_list()[-1])
        #logits = self.fc_2l(agg_res, num_units=[self.hp.d_model, self.hp.num_class], scope="fc_2l")
        return logits

    def _loss_op(self, l2_lambda=0.0001):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.truth))
        weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel') in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
        loss += l2_loss
        return loss

    def _acc_op(self):
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _globalStep_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_op(self):
        # train scheme
        # global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        #optimizer = tf.train.GradientDescentOptimizer(self.hp.lr)
        # optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)

        '''
        if self.hp.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            loss = loss + self.hp.lambda_l2 * l2_loss
        '''

        # grads = self.compute_gradients(loss, tvars)
        # grads, _ = tf.clip_by_global_norm(grads, 10.0)
        # train_op = optimizer.minimize(loss, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(loss)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op

    def predict_model(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y)  # (batchsize, maxlen, d_model)
        x_mask = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)

        # matching
        match_result = match_passage_with_question(x_repre, y_repre, x_mask, y_mask)

        # aggre
        x_inter = self.aggregation(match_result, match_result)  # (?, ?, 512)

        x_avg = tf.reduce_mean(x_inter, axis=1)
        x_max = tf.reduce_max(x_inter, axis=1)

        input2fc = tf.concat([x_avg, x_max], axis=1)
        logits = self.fc(input2fc, match_dim=input2fc.shape.as_list()[-1])

        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(logits, 1, name='label_pred')

        return label_pred
