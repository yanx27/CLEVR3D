import math
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from transformers.models.bert.modeling_bert import (BertEmbeddings, BertEncoder, BertConfig, BertPreTrainedModel)

from transformers.models.lxmert.modeling_lxmert import LxmertCrossAttentionLayer, LxmertSelfAttentionLayer


class TextBert(BertPreTrainedModel):
    def __init__(self, config, mmt_mask=None):
        super().__init__(config)

        self.mmt_mask = mmt_mask
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask, txt_type_mask=None):
        ## https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_bert.html
        encoder_inputs = self.embeddings(txt_inds, token_type_ids=txt_type_mask)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if self.mmt_mask == 'train2dmasklabel':
            to_seq_length = attention_mask.size(1)
            from_seq_length = to_seq_length
            extended_attention_mask = extended_attention_mask.repeat(1, 1, from_seq_length, 1)
            num_query = 24
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(encoder_inputs, extended_attention_mask, head_mask=head_mask)
        seq_output = encoder_outputs[0]
        return seq_output


class AttentionLayer(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, ctx_att_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # if ctx_att_mask is not None:
        #     attention_scores = attention_scores + ctx_att_mask

        # Normalize the attention scores to probabilities.
        # attention_scores = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_scores = self.dropout(attention_scores)

        return attention_scores


class MMT(BertPreTrainedModel):
    def __init__(self, config, context_2d=None, mmt_mask=None, use_cross_att=False):
        super().__init__(config)

        self.context_2d = context_2d
        self.mmt_mask = mmt_mask
        self.encoder = BertEncoder(config)
        self.use_cross_att = use_cross_att
        if self.use_cross_att:
            self.cross_att = LxmertCrossAttentionLayer(config)
            # self.self_att = LxmertSelfAttentionLayer(config)
        self.init_weights()

    def forward(self, txt_emb, txt_mask, obj_emb, obj_mask, obj_num):
        encoder_inputs = torch.cat([txt_emb, obj_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask], dim=1)

        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        txt_begin = 0
        obj_begin = txt_max_num

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length
        extended_attention_mask = extended_attention_mask.repeat(1, 1, from_seq_length, 1)

        # ic(extended_attention_mask.shape, extended_attention_mask[0, 0, 0], extended_attention_mask[0, 0, :, 0])

        if self.mmt_mask == 'train2d' or self.mmt_mask == 'train2dmasklabel':
            # [batch_size, from_seq_length, to_seq_length]
            # mask type 1: 3d, lang can't see 2d

            # decoding step elements can attend to themselves in a causal manner
            num_2d = obj_max_num // 2
            extended_attention_mask[:, :, :-num_2d, -num_2d:] = 0.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(encoder_inputs, extended_attention_mask, head_mask=head_mask, output_attentions=True)

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_max_num]
        mmt_obj_output = mmt_seq_output[:, obj_begin:obj_begin + obj_num]

        if self.use_cross_att:
            # ic(extended_attention_mask.shape)
            # cross_att_mask = extended_attention_mask[..., obj_begin:obj_begin + obj_num, txt_begin:txt_max_num]
            # ic(cross_att_mask.shape)
            cross_att_mask = (obj_mask[:, :obj_num].unsqueeze(2) * txt_mask.unsqueeze(1)).unsqueeze(1)
            cross_att_mask = (1.0 - cross_att_mask) * -10000.0

            cross_output = self.cross_att(
                mmt_obj_output,
                mmt_txt_output,
                ctx_att_mask=cross_att_mask,
                output_attentions=True,
            )

            attentions = cross_output[1].mean(1)

            # attentions = extended_attention_mask[:, 0]
            mmt_obj_output = cross_output[0]

            # mmt_obj_output = self.self_att(
            #     mmt_obj_output, extended_attention_mask[..., obj_begin:obj_begin + obj_num, obj_begin:obj_begin + obj_num])[0]

        else:
            attentions = encoder_outputs['attentions'][-1].mean(1)
            attentions = attentions[:, obj_begin:obj_begin + obj_num, txt_begin:txt_max_num]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_obj_output': mmt_obj_output,
            'cross_attention_output': attentions
        }
        if self.context_2d == 'unaligned':
            results['mmt_obj_output_2D'] = mmt_seq_output[:, obj_begin + obj_num:, :]
        return results


class MatchingLinear(nn.Module):
    def __init__(self, input_size=192, hidden_size=128, outputdim=1):
        super(MatchingLinear, self).__init__()
        hidden_size = input_size * 2 // 3
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, outputdim)

    def forward(self, x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state).squeeze(2)


"""
From VilBert, vilbert/vilbert
"""


class BertLMPredictionHead(nn.Module):
    def __init__(self, bert_model_embedding_weights, input_size=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(input_size=input_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, input_size=None):
        super(BertPredictionHeadTransform, self).__init__()
        hidden_act = "gelu"
        hidden_size = 768
        if input_size is None:
            input_size = hidden_size
        ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.dense = nn.Linear(input_size, hidden_size)
        if isinstance(hidden_act, str) or (sys.version_info[0] == 2 and isinstance(hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class PolluteLinear(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(PolluteLinear, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state)


## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)