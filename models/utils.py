"""
Default choices for auxiliary classifications tasks, encoders & decoders.
"""
import torch.nn as nn
from models.backbone import MLP
from models.backbone.lstm_encoder import LSTMEncoder
from models.backbone import PointNetPP
from models.backbone import Vocabulary

#
# Object Encoder
#


def single_object_encoder(point_dim: int, out_dim: int) -> PointNetPP:
    """
    The default PointNet++ encoder for a 3D object.

    @param: out_dims: The dimension of each object feature
    """
    return PointNetPP(sa_n_points=[32, 16, None],
                      sa_n_samples=[32, 32, None],
                      sa_radii=[0.2, 0.4, None],
                      sa_mlps=[[point_dim, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, out_dim]])


#
#  Text Decoder
#
def text_decoder_for_clf(in_dim: int, n_classes: int) -> MLP:
    """
    Given a text encoder, decode the latent-vector into a set of clf-logits.

    @param in_dim: The dimension of each encoded text feature
    @param n_classes: The number of the fine-grained instance classes
    """
    out_channels = [128, n_classes]
    dropout_rate = [0.2]
    return MLP(in_feat_dims=in_dim, out_channels=out_channels, dropout_rate=dropout_rate)


#
# Object Decoder
#
def object_decoder_for_clf(object_latent_dim: int, n_classes: int) -> MLP:
    """
    The default classification head for the fine-grained object classification.

    @param object_latent_dim: The dimension of each encoded object feature
    @param n_classes: The number of the fine-grained instance classes
    """
    return MLP(object_latent_dim, [128, 256, n_classes], dropout_rate=0.15)


#
#  Token Encoder
#
def token_encoder(word_embedding_dim,
                  lstm_n_hidden: int,
                  word_dropout: float,
                  init_c=None,
                  init_h=None,
                  random_seed=None,
                  feature_type='max'):
    """
    Language Token Encoder.

    @param glove_emb_file: If provided, the glove pretrained embeddings for language word tokens
    @param lstm_n_hidden: The dimension of LSTM hidden state
    @param word_dropout:
    @param init_c:
    @param init_h:
    @param random_seed:
    @param feature_type:
    """

    word_projection = nn.Sequential(nn.Linear(word_embedding_dim, word_embedding_dim), nn.ReLU(), nn.Dropout(word_dropout),
                                    nn.Linear(word_embedding_dim, word_embedding_dim), nn.ReLU())

    model = LSTMEncoder(n_input=word_embedding_dim,
                        n_hidden=lstm_n_hidden,
                        init_c=init_c,
                        init_h=init_h,
                        word_transformation=word_projection,
                        feature_type=feature_type)
    return model, word_projection


#
# Referential Classification Decoder Head
#
def object_lang_clf(in_dim: int, cls: int) -> MLP:
    """
    After the net processes the language and the geometry in the end (head) for each option (object) it
    applies this clf to create a logit.

    @param in_dim: The dimension of the fused object+language feature
    """
    return MLP(in_dim, out_channels=[128, 64, cls], dropout_rate=0.05)


def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def get_siamese_features2(net, in_features):
    n_items = len(in_features)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[i]))
    return out_features