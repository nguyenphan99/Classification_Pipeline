import tensorflow as tf
# from torch.nn import CrossEntropyLoss as CE
def poly1_cross_entropy(logits, labels, epsilon=1.0):
    # pt, CE, and Poly1 have shape [batch].
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1) 
    CE = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1