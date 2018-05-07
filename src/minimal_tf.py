#!/usr/bin/env python
import tensorflow as tf
hello = tf.constant('Hello Lili!')
sess = tf.Session()
print(sess.run(hello))
