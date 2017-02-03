from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from DMNPlus import Config
from DMNPlus import DNMPlus

FLAGS = tf.app.flags.FLAGS

def parse_train_args():
    tf.app.flags.DEFINE_string("b", "1", "babi task 1-20 (default=1)")
    tf.app.flags.DEFINE_boolean("restore", False, "restore previously trained weights (default=false)")
    tf.app.flags.DEFINE_float("l2", 0.001, "specify l2 loss (default=0.001)")

    #don't save earlier epochs - saving weights takes a lot of time in TF
    tf.app.flags.DEFINE_integer("save_from_epoch", 15, "Epoch to save onwards (default=15)")
    tf.app.flags.DEFINE_boolean("train_mode", True, "True if training, False if testing")
    tf.app.flags.DEFINE_integer("epochs", 256, "True if training, False if testing")


def main(argv=None):
    config = Config()
    config.babi_id = FLAGS.b
    config.restore_weights = FLAGS.restore
    config.l2 = FLAGS.l2
    config.epoch_to_save_from = FLAGS.save_from_epoch
    config.train_mode = FLAGS.train_mode
    config.max_epochs = FLAGS.epochs

    # create model
    model = DNMPlus(config)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    model.run_training_or_test(sess)
    sess.close()

if __name__ == '__main__':
    parse_train_args()
    tf.app.run()