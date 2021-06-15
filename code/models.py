from keras.models import Model
from keras.layers import Input

from .layers import GraphAttention, Transition

from keras.regularizers import l1


def graph_model(algo:int, F: int, N: int, f_: int, global_constraint=None):
    """Defines the architecture of the graph neural network"""
    name = 'block1'
    residual_dropout = 0.5
    heads_reduction = 'concat'

    # define the different layers

    attention = GraphAttention(
                 f_,
                 attn_heads=f_,
                 attn_heads_reduction=heads_reduction,  # {'concat', 'average'}
                 dropout_rate=residual_dropout,
                 activation='softmax',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='glorot_uniform',
                 name='attention',
                 attn_kernel_regularizer=l1(0.01),
                 attn_kernel_constraint=global_constraint,
                 use_bias=False, )

    Linear = Transition(
        name=f'{name}_linear1', activation='gelu', size_output=f_, layer_constraint=global_constraint, use_bias=True)

    #########################################
    # Input
    #########################################

    X_in = Input(shape=(F,), name='input1')
    A_in = Input(shape=(N,), name='input2')

    #########################################
    # Model
    #########################################

    # one layer CoBaGAD
    if algo == 0:
        outputs = Linear(X_in)
        outputs = attention([outputs, A_in])
        model = Model(inputs=[X_in, A_in], outputs=outputs)
    else:
        # you can add methods here
        print('Invalid GNN method')

    return model
