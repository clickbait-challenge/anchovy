import time
import datetime
import tensorflow as tf
import numpy as np # scientific computing
import json
import os # to open files
import re # regular expression
import nltk # nlp stuff
from collections import Counter
from sklearn.model_selection import KFold
import emoji

PAD = "<pad>"  # reserve 0 for pad
UNK = "<unk>"  # reserve 1 for unknown
nltk_tokeniser = nltk.tokenize.TweetTokenizer()

class FLAGS:
    dir = "./data"
    training_file = "clickbait17-validation-170630"
    validation_file = "clickbait17-train-170331"
    epochs = 20
    batch_size = 64
    filter_sizes = "3,4,5"
    num_filters = 100
    dropout_rate_hidden = 0.5
    dropout_rate_cell = 0.3
    dropout_rate_embedding = 0.2
    state_size = 64
    hidden_size = 0
    timestamp = "0716"
    y_len = 4
    model = "SAN"
    use_target_description = False
    use_image = False
    learning_rate = 0.01
    embedding_size = 100
    gradient_clipping_value = 1
    if_annotated = 0
    test_file = "clickbait17-train-170331"
    max_target_description_len = 0
    max_post_text_len = 40

# load_embeddings. Opens file fp and does word embedding
def load_embeddings(fp, embedding_size):
    embedding = []
    vocab = []
    with open(fp, 'r', encoding="utf-8") as f:
        for each_line in f:
            row = each_line.split(' ')
            if len(row) == 2:
                continue
            vocab.append(row[0])
            if len(row[1:]) != embedding_size:
                print(row[0])
                print(len(row[1:]))
            embedding.append(np.asarray(row[1:], dtype='float32'))
    word2id = dict(list(zip(vocab, list(range(2, len(vocab))))))
    word2id[PAD] = 0
    word2id[UNK] = 1
    extra_embedding = [np.zeros(embedding_size), np.random.uniform(-0.1, 0.1, embedding_size)]
    embedding = np.append(extra_embedding, embedding, 0)
    return word2id, embedding

# i have no idea why, but it looks important
np.random.seed(81)

word2id, embedding = load_embeddings(fp=os.path.join(FLAGS.dir, "glove.6B."+str(FLAGS.embedding_size)+"d.txt"), embedding_size=FLAGS.embedding_size)
# word2id contains mapping from word to id {'the': 2, ',': 3, ...}
# embedding contains embedding stuff

# writes to word2id.json
with open(os.path.join(FLAGS.dir, 'word2id.json'), 'w', encoding="utf-8") as fout:
        json.dump(word2id, fp=fout)

#utils functions

# AM edit. use twitter tokenize. Delegates, much faster
# T = tokenizer.TweetTokenizer()
# edit not accepted because we need to abstract

#tokenise nltk_tokeniser
def tokenise(text, with_process=True):
    if with_process:
        #return nltk_tokeniser.tokenize(process_tweet(text).lower())
        # print(nltk_tokeniser.tokenize(process_tweet(text).lower()))
        # print(T.tokenize(text)) #not good
        # print("\n")
        return nltk_tokeniser.tokenize(process_tweet(text).lower())
    else:
        # return nltk_tokeniser.tokenize(text)
        return tweet_ark_tokenize(text.lower())

def process_tweet(text):
    FLAGS = re.MULTILINE | re.DOTALL

    def hashtag(text):
        # idea to improve. Build an n-gram model based on a corpus from nltk to order the possibilities by probability of occurence/usage and display only the first
        text = text.group()
        # print(text)
        hashtag_body = text[1:]
        hashtag_body = "".join([a[0].upper()+(a[1:]) for a in re.split("_", hashtag_body)])
        # print(hashtag_body)
        result = " ".join(["<hashtag>"] + re.split(r"([A-Z][a-z]+)", hashtag_body, flags=FLAGS))
        # print(result)
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"

    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    # AM edit
    def re_sub_emojis(str):
        print (str)
        new_str = str
        for c in str:
            if c in emoji.UNICODE_EMOJI:
                # new_str = new_str.replace(c, "<emoji>")
                new_str = new_str.replace(c, emoji.demojize(c))
        print(new_str)
        return new_str

    # print(text)
    text = re_sub_emojis(text)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    # text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    # text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    # text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    # text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    # text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    # pdb.set_trace()
    text = re_sub(r"#\S+", hashtag)
    # text = re.sub(r"#\S+", hashtag, text, flags=FLAGS)
    text = text.replace(r"#\S+", "<hashtag>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)
    # print(text)
    # print("\n")
    return text


#read training data
def read_data(fps, word2id=None, y_len=1, use_target_description=False, use_image=False, delete_irregularities=False):
    ids = []
    post_texts = []
    post_text_lens = []
    truth_means = []
    truth_classes = []
    id2truth_class = {}
    id2truth_mean = {}
    target_descriptions = []
    target_description_lens = []
    image_features = []
    num = 0
    for fp in fps:
        if use_image:
            with open(os.path.join(fp, "id2imageidx.json"), "r", encoding="utf-8") as fin:
                id2imageidx = json.load(fin)
            all_image_features = hickle.load(os.path.join(fp, "image_features.hkl"))
        if y_len:
            with open(os.path.join(fp, 'truth.jsonl'), 'rb', encoding="utf-8") as fin:
                for each_line in fin:
                    each_item = json.loads(each_line.decode('utf-8'))
                    if delete_irregularities:
                        if each_item["truthClass"] == "clickbait" and float(each_item["truthMean"]) < 0.5 or each_item["truthClass"] != "clickbait" and float(each_item["truthMean"]) > 0.5:
                            continue
                    if y_len == 4:
                        each_label = [0, 0, 0, 0]
                        for each_key, each_value in Counter(each_item["truthJudgments"]).items():
                            each_label[int(each_key//0.3)] = float(each_value)/5
                        id2truth_class[each_item["id"]] = each_label
                        if each_item["truthClass"] != "clickbait":
                            assert each_label[0]+each_label[1] > each_label[2]+each_label[3]
                        else:
                            assert each_label[0]+each_label[1] < each_label[2]+each_label[3]
                    if y_len == 2:
                        if each_item["truthClass"] == "clickbait":
                            id2truth_class[each_item["id"]] = [1, 0]
                        else:
                            id2truth_class[each_item["id"]] = [0, 1]
                    if y_len == 1:
                        if each_item["truthClass"] == "clickbait":
                            id2truth_class[each_item["id"]] = [1]
                        else:
                            id2truth_class[each_item["id"]] = [0]
                    id2truth_mean[each_item["id"]] = [float(each_item["truthMean"])]
        with open(os.path.join(fp, 'instances.jsonl'), 'rb', encoding="utf-8") as fin:
            for each_line in fin:
                each_item = json.loads(each_line.decode('utf-8'))
                if each_item["id"] not in id2truth_class and y_len:
                    num += 1
                    continue
                ids.append(each_item["id"])
                each_post_text = " ".join(each_item["postText"])
                each_target_description = each_item["targetTitle"]
                if y_len:
                    truth_means.append(id2truth_mean[each_item["id"]])
                    truth_classes.append(id2truth_class[each_item["id"]])
                if word2id:
                    if (each_post_text+" ").isspace():
                        post_texts.append([0])
                        post_text_lens.append(1)
                    else:
                        each_post_tokens = tokenise(each_post_text)
                        post_texts.append([word2id.get(each_token, 1) for each_token in each_post_tokens])
                        post_text_lens.append(len(each_post_tokens))
                else:
                    post_texts.append([each_post_text])
                if use_target_description:
                    if word2id:
                        if (each_target_description+" ").isspace():
                            target_descriptions.append([0])
                            target_description_lens.append(1)
                        else:
                            each_target_description_tokens = tokenise(each_target_description)
                            target_descriptions.append([word2id.get(each_token, 1) for each_token in each_target_description_tokens])
                            target_description_lens.append(len(each_target_description_tokens))
                    else:
                        target_descriptions.append([each_target_description])
                else:
                    target_descriptions.append([])
                    target_description_lens.append(0)
                if use_image:
                    image_features.append(all_image_features[id2imageidx[each_item["id"]]].flatten())
                else:
                    image_features.append([])
    print("Deleted number of items: " + str(num))
    return ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features


def pad_sequences(sequences, maxlen):
    if maxlen <= 0:
        return sequences
    shape = (len(sequences), maxlen)
    padded_sequences = np.full(shape, 0)
    for i, each_sequence in enumerate(sequences):
        if len(each_sequence) > maxlen:
            padded_sequences[i] = each_sequence[:maxlen]
        else:
            padded_sequences[i, :len(each_sequence)] = each_sequence
    return padded_sequences

# why not eager?
class SAN:
    def __init__(self, x1_maxlen, x2_maxlen, y_len, embedding, filter_sizes, num_filters, hidden_size, state_size, x3_size, attention_size, view_size=1, alpha=0, beta=0):
        if view_size == 1:
            beta = 0
        self.input_x1 = tf.placeholder(tf.int32, [None, x1_maxlen], name="post_text")
        self.input_x1_len = tf.placeholder(tf.int32, [None, ], name="post_text_len")
        self.input_x2 = tf.placeholder(tf.int32, [None, x2_maxlen], name="target_description")
        self.input_x2_len = tf.placeholder(tf.int32, [None, ], name="target_description_len")
        self.input_x3 = tf.placeholder(tf.float32, [None, x3_size], name="image_feature")
        self.input_y = tf.placeholder(tf.float32, [None, y_len], name="truth_class")
        self.input_z = tf.placeholder(tf.float32, [None, 1], name="truth_mean")
        self.dropout_rate_embedding = tf.placeholder(tf.float32, name="dropout_rate_embedding")
        self.dropout_rate_hidden = tf.placeholder(tf.float32, name="dropout_rate_hidden")
        self.dropout_rate_cell = tf.placeholder(tf.float32, name="dropout_rate_cell")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        with tf.variable_scope("embedding"):
            self.W = tf.get_variable(shape=embedding.shape, initializer=tf.constant_initializer(embedding), name="embedding")
            self.embedded_input_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_input_x1 = tf.layers.dropout(self.embedded_input_x1, rate=1-self.dropout_rate_embedding)
        with tf.variable_scope("biRNN"):
            cell_fw = tf.contrib.rnn.GRUCell(state_size)
            cell_dropout_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1-self.dropout_rate_cell)
            initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            cell_bw = tf.contrib.rnn.GRUCell(state_size)
            cell_dropout_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=1-self.dropout_rate_cell)
            initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_dropout_fw, cell_bw=cell_dropout_bw, inputs=self.embedded_input_x1, sequence_length=self.input_x1_len, initial_state_bw=initial_state_bw, initial_state_fw=initial_state_fw)
            bi_outputs = tf.concat(outputs, 2)
        with tf.variable_scope("attention"):
            W_1 = tf.get_variable(shape=[2*state_size, attention_size], initializer=tf.contrib.layers.xavier_initializer(), name="W_1")
            W_2 = tf.get_variable(shape=[attention_size, view_size], initializer=tf.contrib.layers.xavier_initializer(), name="W_2")
            reshaped_bi_outputs = tf.reshape(bi_outputs, shape=[-1, 2*state_size])
            if x3_size:
                # self.compressed_input_x3 = tf.contrib.keras.backend.repeat(tf.layers.dense(tf.layers.dense(self.input_x3, 1024, activation=tf.nn.tanh), attention_size, activation=tf.nn.tanh), x1_maxlen)
                self.compressed_input_x3 = tf.contrib.keras.backend.repeat(tf.layers.dense(self.input_x3, attention_size, activation=tf.nn.tanh), x1_maxlen)
                self.compressed_input_x3 = tf.reshape(self.compressed_input_x3, shape=[-1, attention_size])
                self.attention = tf.nn.softmax(tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(reshaped_bi_outputs, W_1)+self.compressed_input_x3), W_2), shape=[self.batch_size, x1_maxlen, view_size]), dim=1)
            else:
                self.attention = tf.nn.softmax(tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(reshaped_bi_outputs, W_1)), W_2), shape=[self.batch_size, x1_maxlen, view_size]), dim=1)
            attention_output = tf.reshape(tf.matmul(tf.transpose(bi_outputs, perm=[0, 2, 1]), self.attention), shape=[self.batch_size, view_size*2*state_size])
        with tf.variable_scope("penalty"):
            attention_t = tf.transpose(self.attention, perm=[0, 2, 1])
            attention_t_attention = tf.matmul(attention_t, self.attention)
            identity = tf.reshape(tf.tile(tf.diag(tf.ones([view_size])), [self.batch_size, 1]), shape=[self.batch_size, view_size, view_size])
            self.penalised_term = tf.square(tf.norm(attention_t_attention-identity, ord="euclidean", axis=[1, 2]))
        self.h_drop = tf.layers.dropout(attention_output, rate=1-self.dropout_rate_hidden)
        self.scores = tf.layers.dense(inputs=self.h_drop, units=y_len)
        if y_len == 1:
            self.predictions = tf.nn.sigmoid(self.scores, name="prediction")
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.input_z, self.predictions))+beta*self.penalised_term)
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), tf.cast(tf.round(self.input_y), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)+beta*self.penalised_term)
            self.predictions = tf.slice(tf.nn.softmax(self.scores), [0, 0], [-1, 1], name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        elif y_len == 4:
            self.normalised_scores = tf.nn.softmax(self.scores, name="distribution")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)+beta*self.penalised_term)
            self.predictions = tf.matmul(self.normalised_scores, tf.constant([0, 0.3333333333, 0.6666666666, 1.0], shape=[4, 1]), name="prediction")
            self.mse = tf.losses.mean_squared_error(self.input_z, self.predictions)
            correct_predictions = tf.equal(tf.argmax(tf.matmul(self.normalised_scores, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1), tf.argmax(tf.matmul(self.input_y, tf.constant([1, 0, 1, 0, 0, 1, 0, 1], shape=[4, 2], dtype=tf.float32)), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

def get_batch(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    batch_num_per_epoch = int((data_size-1)/batch_size)+1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for i in range(batch_num_per_epoch):
        start_ix = i * batch_size
        end_ix = min((i+1)*batch_size, data_size)
        yield shuffled_data[start_ix:end_ix]
