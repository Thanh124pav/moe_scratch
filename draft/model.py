import tensorflow as tf
from tensorflow.keras import layers, Model

from config import VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, DFF, DROPOUT_RATE, NUM_EXPERTS

# Import MoELayer after it's updated with Hierarchical MoE and Specialization Regularization
from moe_layer import MoELayer

class MoETransformer(Model):
    def __init__(self, vocab_inp_size=VOCAB_SIZE, vocab_tar_size=VOCAB_SIZE, d_model=D_MODEL,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dff=DFF, dropout_rate=DROPOUT_RATE):
        super(MoETransformer, self).__init__()
        
        self.encoder_embedding = layers.Embedding(vocab_inp_size, d_model)
        self.decoder_embedding = layers.Embedding(vocab_tar_size, d_model)
        
        self.positional_encoding = self.positional_encoding(1000, d_model)
        
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                EncoderLayer(d_model, num_heads, dff, dropout_rate)
            )
            # Add MoE layer after every encoder layer
            self.encoder_layers.append(
                MoELayer(d_model, num_experts=NUM_EXPERTS, dff=dff, dropout_rate=dropout_rate)
            )
        
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)
        self.final_layer = layers.Dense(vocab_tar_size)
    
    def positional_encoding(self, length, depth):
        depth = depth/2
        positions = tf.range(length, dtype=tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, ...]
        angle_rates = 1 / (10000** (depths/depth))
        angle_rads = positions * angle_rates
        pos_encoding = tf.concat(
            [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
            axis=-1)
        return pos_encoding
    
    def call(self, inputs, training=True):
        inp, tar = inputs
        
        enc_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_padding_mask, look_ahead_mask)
        
        enc_output = self.encoder_embedding(inp)
        enc_output += self.positional_encoding[tf.newaxis, :tf.shape(inp)[1], :]
        enc_output = self.dropout(enc_output, training=training)
        
        load_balance_loss = 0.0
        specialization_loss = 0.0
        for i in range(0, len(self.encoder_layers), 2):
            enc_output = self.encoder_layers[i](enc_output, training, enc_padding_mask)
            moe_output, routing_outputs = self.encoder_layers[i+1](enc_output, training=training)
            enc_output = moe_output
            if training:
                load_balance_loss += self.encoder_layers[i+1].get_load_balance_loss(routing_outputs)
                specialization_loss += self.encoder_layers[i+1].get_specialization_loss()
        
        dec_output = self.decoder_embedding(tar)
        dec_output += self.positional_encoding[tf.newaxis, :tf.shape(tar)[1], :]
        dec_output = self.dropout(dec_output, training=training)
        
        for i in range(len(self.decoder_layers)):
            dec_output = self.decoder_layers[i](dec_output, enc_output, training, combined_mask, enc_padding_mask)
        
        final_output = self.final_layer(dec_output)
        
        return final_output, load_balance_loss, specialization_loss
    
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        
        return output

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    
    return output

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ]) 