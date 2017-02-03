from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import babi_input
from Attention import DMNAttentionGate
from AttentionGRU import AttentionGRU

from dnn import l2_reg
from dnn import random_uniform_init

import sys
import os
import time

class Config(object):

    # used in load_babi: word2vec_init, embed_size, floatX, embedding_init, train_mode, max_allowed_inputs, num_train
    # foll params are added to config in the code after init : max_words_per_question, max_sentences_per_input, max_words_per_sentence
    batch_size = 128
    embed_size = 80
    hidden_size = 80
    max_epochs = 256
    early_stopping = 20
    dropout_keep_prob = 0.9
    lr = 0.001
    l2 = 0.001

    word2vec_init = False #not used (no glove or word_to_vec)
    embedding_init = 1.7320508  # not used, just kept for babi_input script to work

    dropout_in_gru_rnn = False
    num_episodes = 3
    num_attention_features = 4
    dtype=np.float32
    max_allowed_inputs = 130 #From paper: we limited the input to the last 70 sentences for all tasks except QA3 for which we limited input to the last 130 sentences
    num_train = 9000

    babi_id = "1"
    babi_test_id = ""
    train_mode = True
    restore_weights = False

    #log summary every nth step
    summary_log_step = 10

    num_gpu = None #incase of multi gpu training (not impl in the current version)
    epoch_to_save_from = 25 #don't save earlier epochs - saving weights takes a lot of time in TF for large graphs


def _position_encoding(sentence_size, embedding_size):
    # from https://github.com/domluna/memn2n
    #from author: fix positional encoding so that it varies according to sentence lengths (?)
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def add_word_embedding_layer(vocab_size, embed_size, initializer=None):
    #dim: vocab_size, embed_size
    W_embed = tf.get_variable("W_embed", shape=[vocab_size, embed_size], dtype=tf.float32,
                              initializer=initializer, regularizer=l2_reg, trainable=True)
    return W_embed


def embedded_lookup_op(W_embed, inputs):
    #w_embed dim: vocab_size, embed_size
    #inputs
    #   inputs : dim: batch_size, max_sentences_per_input, max_words_per_sentence
    #   questions: dim: batch_size, max_words_per_question

    # embedded_lookup_matrix (for inputs) dim: batch_size, max_sentences_per_input, max_words_per_sentence, embed_size
    # embedded_lookup_matrix (for questions) dim: batch_size, max_words_per_question, embed_size
    embedded_lookup_matrix = tf.nn.embedding_lookup(W_embed, inputs)
    return embedded_lookup_matrix


def add_gru_layer(inputs, hidden_size, len_per_input_ph, dropout_in_gru_rnn, dropout_keep_prob_ph):
    gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    # apply droput to grus if flag set
    if dropout_in_gru_rnn:
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=dropout_keep_prob_ph,
                                                      output_keep_prob=dropout_keep_prob_ph)

    outputs, final_state = tf.nn.dynamic_rnn(gru_cell, tf.stack(inputs, axis=0), dtype=np.float32, sequence_length=len_per_input_ph, time_major=True)
    return outputs, final_state

def add_bi_dirn_gru_layer(inputs, hidden_size, len_per_input_ph, dropout_in_gru_rnn, dropout_keep_prob_ph):
    gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    # apply droput to grus if flag set
    if dropout_in_gru_rnn:
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=dropout_keep_prob_ph,
                                                      output_keep_prob=dropout_keep_prob_ph)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, inputs, dtype=np.float32,sequence_length=len_per_input_ph)
    final_state_fw, final_state_bw = states
    outputs_fw, outputs_bw = outputs
    return outputs_fw, outputs_bw, final_state_fw, final_state_bw


def add_answers(memory_outputs, questions, vocab_size, hidden_size, dropout_keep_prob_ph, initializer=tf.contrib.layers.xavier_initializer()):

    #The paper mentions droput is applied at answer modeule - since we can't apply it at the answer_logits, we are applying here
    #batch_size, hidden_size
    memory_outputs = tf.nn.dropout(memory_outputs, dropout_keep_prob_ph)

    W_answer = tf.get_variable("U", (2 * hidden_size, vocab_size), initializer=initializer, regularizer=l2_reg)
    b_answer = tf.get_variable("bias_p", (vocab_size,), initializer=tf.constant_initializer(0.0))

    #batch_size, vocab_size
    answer_logits = tf.matmul(tf.concat(1, [memory_outputs, questions]), W_answer) + b_answer

    return answer_logits


def add_loss_op(answers_logits, answers, l2):
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(answers_logits, answers))
    reg_losses = l2 * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss += reg_losses
    return loss


def add_training_op(loss, lr):
    """Calculate and apply gradients"""

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    return train_op


def add_predictions(output):
    preds = tf.nn.softmax(output)
    pred = tf.argmax(preds, 1)
    return pred


def build_dnm_plus_graph(config, qbt_data, training_graph=True):
    if training_graph:
        summary_str = "_Train"
    else:
        summary_str = "_Valid"


    # dim: scalar
    dropout_keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout_keep_prob_placeholder')

    """
    questions_qbt: dim: batch_size, max_words_per_question,
                   dtype: tf.int32
    inputs_qbt: Each input has multiple sentences, however there is only one answer and one question associated with it
                dim: batch_size, max_sentences_per_input, max_words_per_sentence
                dtype: int32

    words_per_question_in_batch_qbt:
                dim: batch_size
                dtype: tf.int32

    sentences_per_input_in_batch_qbt:
                dim: batch_size
                dtype: tf.int32

    answers_qbt: assuming single word answer.
                dim: batch_size
                dtype: tf.int64: because tf.argmax returns int64: and mismatch exception at tf.contrib accuracy()

    #Use this if there are multiple words per answer:
                shape None, max_words_per_answer
                dtype: tf.int32
    """

    (questions_qbt, inputs_qbt, words_per_question_in_batch_qbt, sentences_per_input_in_batch_qbt,
     answers_qbt) = qbt_data

    with tf.variable_scope("WordEmbed"):
        # Add Embedding
        W_embed = add_word_embedding_layer(config.vocab_size, config.embed_size, initializer=random_uniform_init(tf.sqrt(3.0))) #,initializer=tf.contrib.layers.xavier_initializer())

        #inputs dim: batch_size, max_sentences_per_input, max_words_per_sentence, embed_size
        inputs = embedded_lookup_op(W_embed, inputs_qbt)

    with tf.variable_scope("Question"):
        # questions dim: [batch_size, max_words_per_question, embed_size]
        questions = embedded_lookup_op(W_embed, questions_qbt)
        # output dim for questions: [max_words_per_question, [batch_size, embed_size]]
        questions = tf.unpack(questions, axis=1)

        # Add question layer: encode the question using a GRU
        #questions_gru_final_state dim: batch_size, hidden_size

        _, questions_gru_final_state = add_gru_layer(questions, config.hidden_size, words_per_question_in_batch_qbt,
                                                  config.dropout_in_gru_rnn, dropout_keep_prob_placeholder)

    with tf.variable_scope("InputModule"):
        with tf.variable_scope("PositionalEncoding"):
            # Input fusion layer
            # Add sentence encoder: encode sentences into facts using positional encoder
            posn_encoding = tf.constant(_position_encoding(config.max_words_per_sentence, config.embed_size), dtype=tf.float32,name="posn_encoding")
            #posn_encoding = _position_encoding(config.max_words_per_sentence, config.embed_size)
            #apply posn encoding
            # dim :batch_size, max_sentences_per_input, embed_size
            inputs = tf.reduce_sum(inputs * posn_encoding, 2)

        with tf.variable_scope("InputFusionLayer"):
            # Dynamic bidirn rnn: facts_from_fusion_layer dim: for fw and bw each: [batch_size, max_sentences_per_input, hidden_size]
            facts_from_fusion_layer_fw, facts_from_fusion_layer_fw, _, _ = add_bi_dirn_gru_layer(inputs, config.hidden_size,
                                                                  sentences_per_input_in_batch_qbt,
                                                                  config.dropout_in_gru_rnn, dropout_keep_prob_placeholder)

            facts_from_fusion_layer = tf.add(facts_from_fusion_layer_fw, facts_from_fusion_layer_fw)
            #[max_sentences_per_input, [batch_size, hidden_size]]
            facts_from_fusion_layer = tf.unstack(facts_from_fusion_layer, axis=1)

            #Note: dropout can be optimized by operating on tensor directly instead of below looping
            facts_from_fusion_layer = [tf.nn.dropout(fv, dropout_keep_prob_placeholder) for fv in facts_from_fusion_layer]

    # Add Episodic Memory Layer: Attention GRU's, Episodic Memories
    with tf.variable_scope("EpisodicMemoryModule"):
        with tf.variable_scope("AttentionGate"):
            attn_gate = DMNAttentionGate(config.hidden_size, config.num_attention_features)
        with tf.variable_scope("AttentionGRU"):
            attn_gru = AttentionGRU(config.hidden_size)

        attn_gru_prev_state = questions_gru_final_state
        attn_gru_output = None

        curr_memory = None
        memory_unit_prev_state = questions_gru_final_state

        for i in range(config.num_episodes):
            with tf.name_scope("EpisodicMemoryPass" + str(i)):
                #attentions dim: max_sentences_per_input, batch_size
                attentions = attn_gate.get_attention(questions_gru_final_state, memory_unit_prev_state, facts_from_fusion_layer, sentences_per_input_in_batch_qbt)

                """
                # Alternate Attn GRU impl: not working

                reuse = False if i == 0 else True
                attn_gru_cell = AttnGRUCell(config.hidden_size, attentions, reuse=reuse)

                # batch_size, hidden_size
                _, attn_gru_output = tf.nn.dynamic_rnn(attn_gru_cell, tf.stack(facts_from_fusion_layer, axis=0), dtype=np.float32,
                                                         initial_state= attn_gru_prev_state, time_major=True)
                attn_gru_prev_state = attn_gru_output
                """

                for j in range(config.max_sentences_per_input):
                    #convert (batch_size,) to (batch_size,1): needed for matmul
                    attn_for_curr_fact = tf.expand_dims(attentions[j], axis=1)
                    #batch_size, hidden_size
                    attn_gru_output = attn_gru(facts_from_fusion_layer[j], attn_gru_prev_state, attn_for_curr_fact)
                    attn_gru_prev_state = attn_gru_output

                with tf.variable_scope("EpisodicMemoryUpdate"):
                    # untied weights for memory update
                    Wt = tf.get_variable("W_t" + str(i), (3 * config.hidden_size, config.hidden_size),
                                         initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_reg)
                    bt = tf.get_variable("bias_t" + str(i), (config.hidden_size,), initializer=tf.constant_initializer(0.0))
                    # update memory with Relu
                    curr_memory = tf.nn.relu(tf.matmul(tf.concat(1, [memory_unit_prev_state, attn_gru_output, questions_gru_final_state]), Wt) + bt)
                memory_unit_prev_state = curr_memory

    # Add Answer Layer
    with tf.variable_scope("Answer"):
        answers_logits = add_answers(curr_memory, questions_gru_final_state, config.vocab_size, config.hidden_size, dropout_keep_prob_placeholder)

    with tf.variable_scope("Prediction"):
        pred = add_predictions(answers_logits)
        accuracy = tf.contrib.metrics.accuracy(pred, answers_qbt)
        acc_summary = tf.summary.scalar('Accuracy'+summary_str, accuracy)

    #add loss
    with tf.variable_scope("Loss"):
        loss = add_loss_op(answers_logits, answers_qbt, config.l2)
        loss_summary = tf.summary.scalar('Loss'+summary_str, loss)

    # Add Training Op
    with tf.variable_scope("TrainingOp"):
        if training_graph:
            train_op = add_training_op(loss, config.lr)
        else:
            train_op = tf.no_op()

    # Add Summary Op
    #do not use tf.summary.merge_all() - incase of validation it will pull all
    summary_op = tf.summary.merge([acc_summary, loss_summary])

    #self.tr_dropout_ph, self.tr_loss, self.tr_pred, self.tr_accuracy, self.tr_train_op
    return dropout_keep_prob_placeholder, loss, pred, accuracy, train_op, summary_op


class DNMPlus(object):
    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        #num_supp_facts: not used (Strong supervised training)
        #note: word_embedding returned by load_babi is not used - instead we define it using TF's functions
        if self.config.train_mode:
            self.train_or_test_data, self.valid_data, _, self.config.max_words_per_question, self.config.max_sentences_per_input, \
            self.config.max_words_per_sentence, self.config.num_supp_facts, self.config.vocab_size \
                = babi_input.load_babi(self.config, split_sentences=True)
        else:
            self.train_or_test_data, _, self.config.max_words_per_question, self.config.max_sentences_per_input, \
            self.config.max_words_per_sentence, self.config.num_supp_facts, self.config.vocab_size \
                = babi_input.load_babi(self.config, split_sentences=True)


    def create_q_based_tensors(self, data, batch_size, max_epochs=1, shuffle=True):
        questions, inputs, words_per_question_in_batch, sentences_per_input_in_batch, _, answers, _ = data
        questions = tf.constant(questions, dtype=tf.int32)
        inputs = tf.constant(inputs, dtype=tf.int32)
        words_per_question_in_batch = tf.constant(words_per_question_in_batch, dtype=tf.int32)
        sentences_per_input_in_batch = tf.constant(sentences_per_input_in_batch, dtype=tf.int32)
        answers = tf.constant(answers, dtype=tf.int64)

        questions, inputs, words_per_question_in_batch, sentences_per_input_in_batch, answers = \
            tf.train.slice_input_producer([questions, inputs, words_per_question_in_batch,
                                           sentences_per_input_in_batch, answers], num_epochs=max_epochs, shuffle=shuffle)
        #optionally we can set allow_smaller_final_batch=True
        questions, inputs, words_per_question_in_batch, sentences_per_input_in_batch, answers = \
            tf.train.batch([questions, inputs, words_per_question_in_batch, sentences_per_input_in_batch, answers],
                           batch_size=batch_size)
        return questions, inputs, words_per_question_in_batch, sentences_per_input_in_batch, answers

    def __init__(self, config):
        self.config = config
        # adds to self: vocab, encoded_train, encoded_valid, encoded_test
        self.load_data(debug=False)

        #for testing we need to compute only one epoch
        if not config.train_mode:
            config.max_epochs = 1
            config.dropout_keep_prob = 1.0
            #turn off shuffle for testing!
            shuffle = False
        else:
            shuffle = True

        self.train_test_batches_per_epoch = len(self.train_or_test_data[0]) // config.batch_size

        with tf.variable_scope("DMN"):
            with tf.variable_scope("InputData"): #name_Scope would have been fine - there are no variables here
                with tf.variable_scope("Train"):
                    #for train or test
                    questions_qbt_train, inputs_qbt_train, words_per_question_in_batch_qbt_train, sentences_per_input_in_batch_qbt_train, answers_qbt_train = \
                        self.create_q_based_tensors(self.train_or_test_data, batch_size=config.batch_size, max_epochs=config.max_epochs, shuffle=shuffle)

                    qbt_train = (questions_qbt_train, inputs_qbt_train, words_per_question_in_batch_qbt_train,
                                 sentences_per_input_in_batch_qbt_train, answers_qbt_train)

                with tf.variable_scope("Valid"):
                    #setup validation data only in training mode
                    if config.train_mode:
                        self.valid_batches_per_epoch = len(self.valid_data[0]) // config.batch_size
                        questions_qbt_valid, inputs_qbt_valid, words_per_question_in_batch_qbt_valid, sentences_per_input_in_batch_qbt_valid, answers_qbt_valid = \
                            self.create_q_based_tensors(self.valid_data, batch_size=config.batch_size, max_epochs=config.max_epochs,
                                                    shuffle=False)
                        qbt_valid = (questions_qbt_valid, inputs_qbt_valid, words_per_question_in_batch_qbt_valid,
                                     sentences_per_input_in_batch_qbt_valid, answers_qbt_valid)

            #Build the graph!

            with tf.name_scope("Train"):
                with tf.variable_scope("Model"):
                    start_time = time.time()
                    self.tr_dropout_ph, self.tr_loss, self.tr_pred, self.tr_accuracy, self.tr_train_op, self.tr_summary_op = \
                        build_dnm_plus_graph(config, qbt_train, training_graph=True)
                    print("--- Training Graph Building Time: %s seconds ---" % (time.time() - start_time))

            if config.train_mode:
                with tf.name_scope("Valid"):
                    with tf.variable_scope("Model") as model_scope:
                        model_scope.reuse_variables()
                        start_time = time.time()
                        self.val_dropout_ph, self.val_loss, self.val_pred, self.val_accuracy, self.val_train_op, self.val_summary_op = \
                            build_dnm_plus_graph(config, qbt_valid, training_graph=False)
                        print("--- Valid Graph Building Time: %s seconds ---" % (time.time() - start_time))

            """
            #Same utility as above but using make_template: This also works.
            model = tf.make_template("Model", build_dnm_plus_graph)

            with tf.name_scope("Train"):
                start_time = time.time()
                self.tr_dropout_ph, self.tr_loss, self.tr_pred, self.tr_accuracy, self.tr_train_op, self.tr_summary_op = model(config, qbt_train, training_graph=True)
                print("--- %s seconds ---" % (time.time() - start_time))

            with tf.name_scope("Valid"):
                if config.train_mode:
                    start_time = time.time()
                    self.val_dropout_ph, self.val_loss, self.val_pred, self.val_accuracy, self.val_train_op, self.val_summary_op = model(config, qbt_valid, training_graph=False)
                    print("--- %s seconds ---" % (time.time() - start_time))
            """
    def run_training_or_test(self, session, verbose=2):

        config = self.config

        #summary writer
        summaries_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)
        train_writer = tf.summary.FileWriter(summaries_dir, session.graph)

        #Setup the saver
        saver = tf.train.Saver()
        weights_dir = 'weights/task'
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        #restore weights if needed (or if in testing mode)
        if not config.train_mode or config.restore_weights:
            print('Restoring weights...')
            saver.restore(session, 'weights/task' + str(config.babi_id) + '.weights')

        dp_valid = 1.0

        best_val_epoch = 0
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        total_loss = []
        accuracy = 0.0
        epoch_num = 0

        print("Begin Train\n") if config.train_mode else print("Begin Test\n")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                start = time.time()
                feed = {self.tr_dropout_ph: config.dropout_keep_prob}
                loss, acc, summary, _ = session.run([self.tr_loss, self.tr_accuracy, self.tr_summary_op, self.tr_train_op], feed_dict=feed)
                accuracy += acc
                step += 1

                if step % config.summary_log_step == 0:
                    train_writer.add_summary(summary, step)

                total_loss.append(loss)
                if verbose and step % verbose == 0:
                    sys.stdout.write('\r{}  : loss = {}'.format(step, np.mean(total_loss)))
                    sys.stdout.flush()

                # for each epoch
                if step % self.train_test_batches_per_epoch == 0:
                    print("\r                                              ")
                    print("Epoch: ", epoch_num)
                    print("Loss: ", np.mean(total_loss))
                    print("Accuracy: ", accuracy / float(self.train_test_batches_per_epoch))

                    # compute validation if in training mode
                    if config.train_mode:
                        val_step = 0
                        val_accuracy_total = 0.0
                        val_total_loss = []
                        try:
                            while not coord.should_stop():
                                feed = {self.val_dropout_ph: dp_valid}
                                val_loss_batch, val_acc, val_summary = session.run([self.val_loss, self.val_accuracy, self.val_summary_op],
                                                           feed_dict=feed)
                                val_accuracy_total += val_acc
                                val_step += 1

                                val_total_loss.append(val_loss_batch)
                                if verbose and step % verbose == 0:
                                    sys.stdout.write('\r{}  : loss = {}'.format(val_step, np.mean(val_total_loss)))
                                    sys.stdout.flush()

                                # run validation only for one epoch of validation data
                                if val_step % self.valid_batches_per_epoch == 0:
                                    break
                        except tf.errors.OutOfRangeError:
                            print("Val Done: ", val_step)
                        val_loss = np.mean(val_total_loss)
                        val_accuracy = val_accuracy_total/float(self.valid_batches_per_epoch)
                        print("\rValid loss: ", val_loss)
                        print("\rValid accuracy: ", val_accuracy)
                        train_writer.add_summary(val_summary, step) #note: this just stores last mini-batch's summary - not the mean
                        #print("\rValid accuracy: ", val_accuracy / float(self.valid_batches_per_epoch))

                        #save the weights if this validation loss is lesser than the best
                        if val_loss < best_val_loss:
                            best_val_epoch = epoch_num
                            best_val_loss = val_loss
                            best_val_accuracy = val_accuracy
                            print('*************Saving weights****************')
                            if epoch_num < config.epoch_to_save_from:
                                print("Not saving earlier epochs")
                            else:
                                saver.save(session, './weights/task' + str(config.babi_id) + '.weights')

                        # set/reset train counters for next epoch
                        epoch_num += 1
                        total_loss = []
                        accuracy = 0

                    #check for early stopping
                    if epoch_num - best_val_epoch > config.early_stopping:
                        break

                    #todo: maybe try annealing learning rate esp for task id 3

                    #time taken for epoch
                    print('Total time: {}'.format(time.time() - start))
        except tf.errors.OutOfRangeError:
            print("Done: ", step)

        finally:
            # When done, request threads to stop
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        train_writer.close()

        if config.train_mode:
            print("Best val epoch: ", best_val_epoch, " , best val accuracy: ", best_val_accuracy)
        print("Complete")
