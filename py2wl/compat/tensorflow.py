"""
compat/tensorflow.py — TensorFlow/Keras 兼容代理
用法：
    from py2wl.compat import tensorflow as tf
    out = tf.linalg.det(matrix)
"""
import sys
from ._proxy_base import LibraryProxy

sys.modules[__name__] = LibraryProxy("tf")
