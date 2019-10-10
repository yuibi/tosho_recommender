import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, Bidirectional, Concatenate, BatchNormalization, Input
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model
import numpy

class GRU_with_Attn(layers.Layer):
    """
    Reference
    ---------
    A Structured Self-attentive Sentence Embedding (Lin, et al., 2017) (https://arxiv.org/abs/1703.03130)
    Parameters
    ----------
        rnn_unit : ``int``, required.
            Number of RNN unit. Since this is bidirectional, the output shape will be rnn_unit * 2
        dropout_rate : ``float``. optional (default = 0.3).
            Dropout rate.
        concat_list : ``bool``, optional (default = False).
            If the input is single or multiple (e.g. a list of book representations).
        nch : ``int``, optional (default = 24).
            Units for linear layer used to calculate self-attention weights
        Returns
        -------
        x : ``tensorflow.python.framework.ops.EagerTensor``
            Self-attention weights.
            shape=(batch_size, seq_len, 1)
            e.g. array([[[0.01],[0.02],...]])
        y : ``tensorflow.python.framework.ops.EagerTensor``
            Sum of the GRU hidden states according to the weight provided by the self-attention weights.
            shape=(batch_size, rnn_unit * 2)
            e.g. array([[0.01, 0.02, ...]])
    """
    def __init__(self,
                 rnn_unit:int=None,
                 dropout_rate:float=.3,
                 concat_list:bool=False,
                 nch:int=24, **kwargs
                ):
        super(GRU_with_Attn, self).__init__()
        self.concat_list = concat_list
        self.concat = Concatenate(axis=-1)
        self.gru = Bidirectional(GRU(
                              units=rnn_unit,
                              dropout=dropout_rate,
                              return_sequences=True))
        
        self.main = Sequential()
        self.main.add(Dense(nch, activation="relu"))
        self.main.add(Dense(1))

    def call(self,
             inputs  # (batch_size, seq_len, hidden_sz)
            ):
        
        if self.concat_list:
            hidden_size = inputs[0].shape[1]
            num_inputs = len(inputs)
            inputs = self.concat(inputs)
            inputs = tf.reshape(inputs, [-1, num_inputs, hidden_size])
            
        inputs = self.gru(inputs)
        x = self.main(inputs)
        x = tf.nn.softmax(x, axis=1) # (b, s, h) -> (b, s, 1)
        y = tf.reduce_sum((inputs * x), axis=1) # (b, h)
        return [x, y]

def gru_model(
                 embedding_dim:int=None,
                 dropout_rate:float=.3,
                 rnn_unit:int=None,
                 input_shape:tuple=None,
                 num_features:int=None,
                 share_gru_weights_on_book:bool=True,
                 use_attention_on_book:bool=False,
                 use_attention_on_user:bool=True,
                 use_batch_norm:bool=False,
                 is_embedding_trainable:bool=False,
                 final_activation:str='tanh',
                 final_dimension:int=300,
                 embedding_matrix:numpy.ndarray=None):
    """
    Parameters
    ----------
        embedding_dim : ``int``, required.
            Number of word embedding dimension. e.g. 300 
        dropout_rate : ``float``. optional (default = 0.3).
            Dropout rate.
        rnn_unit : ``int``, required.
            Number of RNN unit. Since this is bidirectional, the output shape will be rnn_unit * 2
        input_shape : ``tuple``. required.
            Sequence length in tuple. e.g. (500,)
        num_features : ``int``. required.
            Number of word features. e.g. 20001
        share_gru_weights_on_book : ``bool``, optional (default = True).
            If book-level GRU weights are shared.
        use_attention_on_book : ``bool``, optional (default = False).
            If self-attention is used for book-level.
        use_attention_on_user : ``bool``, optional (default = True).
            If self-attention is used for user-level (=a series of books).
        use_batch_norm : ``bool``, optional (default = False).
            If batch normalization is used.
        is_embedding_trainable : ``bool``, optional (default = False).
            If word embedding layer is trainable.
        final_activation : ``str``, optional (default = 'tanh').
            Activation used for the final linear layer.
        final_dimension : ``int``, optional (default = 300).
            Output dimension of the final linear layer.
        embedding_matrix : ``numpy.ndarray``. required.
            Pretrained word vector weights.
            shape=(num_features, 300)
            e.g. array([[0.01, 0.02, ..],[0.01, 0.02, ..],...])
        Returns
        -------
        x : ``tensorflow.python.framework.ops.EagerTensor``
            Self-attention weights.
            shape=(batch_size, seq_len, 1)
            e.g. array([[[0.01],[0.02],...]])
        y : ``tensorflow.python.framework.ops.EagerTensor``
            Sum of the GRU hidden states according to the weight provided by the self-attention weights.
            shape=(batch_size, rnn_unit * 2)
            e.g. array([[0.01, 0.02, ...]])
    """
    word_id_1 = Input(shape=(None,))
    word_id_2 = Input(shape=(None,))
    word_id_3 = Input(shape=(None,))
    word_id_4 = Input(shape=(None,))

    # Word embedding
    word_emb_1 = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable,
                            name='word_embed_1',
                            mask_zero=True,
                            )(word_id_1)
    word_emb_2 = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable,
                            name='word_embed_2',
                            mask_zero=True,
                            )(word_id_2)
    word_emb_3 = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable,
                            name='word_embed_3',
                            mask_zero=True,
                            )(word_id_3)
    word_emb_4 = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable,
                            name='word_embed_4',
                            mask_zero=True,
                            )(word_id_4)
    
    if use_batch_norm:
        word_emb_1 = BatchNormalization()(word_emb_1)
        word_emb_2 = BatchNormalization()(word_emb_2)
        word_emb_3 = BatchNormalization()(word_emb_3)
        word_emb_4 = BatchNormalization()(word_emb_4)

    if share_gru_weights_on_book:
        if use_attention_on_book:
            gru = GRU_with_Attn(rnn_unit, dropout_rate, concat_list=False)
            gru1 = gru(word_emb_1)
            gru1 = gru1[1]
            gru2 = gru(word_emb_2)
            gru2 = gru2[1]
            gru3 = gru(word_emb_3)
            gru3 = gru3[1]
            gru4 = gru(word_emb_4)
            gru4 = gru4[1]
        else:
            gru = Bidirectional(GRU(
                                units=rnn_unit,
                                dropout=dropout_rate,
                                return_sequences=False))
            gru1 = gru(word_emb_1)
            gru2 = gru(word_emb_2)
            gru3 = gru(word_emb_3)
            gru4 = gru(word_emb_4)
    else:
        if use_attention_on_book:
            gru1 = GRU_with_Attn(rnn_unit, dropout_rate, concat_list=False)(word_emb_1)
            gru1 = gru1[1]
            gru2 = GRU_with_Attn(rnn_unit, dropout_rate, concat_list=False)(word_emb_2)
            gru2 = gru2[1]
            gru3 = GRU_with_Attn(rnn_unit, dropout_rate, concat_list=False)(word_emb_3)
            gru3 = gru3[1]
            gru4 = GRU_with_Attn(rnn_unit, dropout_rate, concat_list=False)(word_emb_4)
            gru4 = gru4[1]
        else:
            gru1 = Bidirectional(GRU(
                                units=rnn_unit,
                                dropout=dropout_rate,
                                return_sequences=False))(word_emb_1)
            gru2 = Bidirectional(GRU(
                                units=rnn_unit,
                                dropout=dropout_rate,
                                return_sequences=False))(word_emb_2)
            gru3 = Bidirectional(GRU(
                                units=rnn_unit,
                                dropout=dropout_rate,
                                return_sequences=False))(word_emb_3)
            gru4 = Bidirectional(GRU(
                                units=rnn_unit,
                                dropout=dropout_rate,
                                return_sequences=False))(word_emb_4)

    if use_batch_norm:
        gru1 = BatchNormalization()(gru1)
        gru2 = BatchNormalization()(gru2)
        gru3 = BatchNormalization()(gru3)
        gru4 = BatchNormalization()(gru4)

    if use_attention_on_user:
        x = GRU_with_Attn(rnn_unit, dropout_rate, concat_list=True)([gru1, gru2, gru3, gru4])
        x = x[1]
    else:
        x = Concatenate(axis=-1)([gru1, gru2, gru3, gru4])
        hidden_size = gru1.shape[1]
        num_inputs = len([gru1, gru2, gru3, gru4])
        x = tf.reshape(x, [-1, num_inputs, hidden_size])
        x = Bidirectional(GRU(
                                units=rnn_unit,
                                dropout=dropout_rate,
                                return_sequences=False))(x)
    
    pred = Dense(final_dimension, activation=final_activation, name='final_dense')(x)
    
    model = Model(inputs=[word_id_1, word_id_2, word_id_3, word_id_4], outputs=pred)
    
    return model