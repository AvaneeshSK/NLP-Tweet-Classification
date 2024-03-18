# Tweet Classification -> disaster or not
    # GRU had better scores
# https://www.kaggle.com/competitions/nlp-getting-started

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.naive_bayes import MultinomialNB
# tfidf works by assinging weights to words, words that appear more often are assigned with higher weights and vice versa
# so, looking at output of tfidf we see values that are closer to 0 are the words that appear less often and vice versa
from sklearn.feature_extraction.text import TfidfVectorizer # Term Frequency - Inverse Document Frequency

df_ = pd.read_csv('Machine Learning 3/nlp-getting-started/train.csv')
df = df_.copy()[['text', 'target']]

# value counts of target 1, 0
# fig, ax = plt.subplots()
# df['target'].value_counts().plot.bar(color=['tomato', 'lightblue'], ax=ax)
# plt.show()

X = df['text'].to_numpy()
y = df['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# check lengths of words distribution :  seems like most words lenghts are between 120 and 140
# lengths_of_each_word = []
# for each in X_train:
#     lengths_of_each_word.append(len(each))
# plt.hist(lengths_of_each_word, bins=25)
# plt.show()

# find average word length
total_length = 0
for word in X:
    total_length += len(word.split())
avg_word_length = int(np.round(total_length/len(X)))

# Modelling : 
    # Tokenization - converting words to int
text_vectorization_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=15000,
    ngrams=None,
    output_mode='int',
    output_sequence_length=avg_word_length,
    split='whitespace',
    standardize='lower_and_strip_punctuation',
    pad_to_max_tokens=True
)
text_vectorization_layer.adapt(X_train)
    # Embedding -  forming relation among data 
embedding_layer = tf.keras.layers.Embedding(
    input_length=avg_word_length, 
    output_dim=128,
    input_dim=text_vectorization_layer.vocabulary_size() # 15000
)

# model 1 : RNN LSTM
input_layer = tf.keras.layers.Input(shape=(1, ), dtype='string')
x = text_vectorization_layer(input_layer)
y = embedding_layer(x)
# return sequences is True if we want to chain together lstm layers
lstm1 = tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True)(y) # weights are readjusted on this layer by Adam()
lstm2 = tf.keras.layers.LSTM(units=64, activation='tanh')(lstm1) # weights are readjusted here
output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(lstm2) # just linear outputs from the lstm layer no additional weights here

rnn_model = tf.keras.Model(input_layer, output_layer)

rnn_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

rnn_model.fit(
    x=X_train,
    y=y_train,
    verbose=2,
    epochs=10,
    batch_size=128,
    shuffle=True
)

preds = rnn_model.predict(X_test)
preds_int = []
for pred in preds.flatten():
    preds_int.append(int(np.round(pred)))
rnn_model_score = accuracy_score(y_pred=preds_int, y_true=y_test)
rnn_cfmatrix = confusion_matrix(y_pred=preds_int, y_true=y_test)

# model 2 : MultinomialNB
tfidf = TfidfVectorizer()
transformed_X_train = tfidf.fit_transform(X_train) # transform is for training data, we need learnt weights from X_train
transformed_X_test = tfidf.transform(X_test) # we dont need to fit since we dont want learnt weights from X_test

sklearn_model = MultinomialNB()

sklearn_model.fit(transformed_X_train, y_train)

preds = sklearn_model.predict(transformed_X_test)
sklearn_model_score = accuracy_score(y_pred=preds, y_true=y_test)
sklearn_cfmatrix = confusion_matrix(y_pred=preds, y_true=y_test)

# model 3 : CNN 1D

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    text_vectorization_layer,
    embedding_layer,
    tf.keras.layers.Conv1D(
        filters=64,
        activation='relu',
        strides=1,
        kernel_size=3
    ),
    tf.keras.layers.MaxPool1D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Conv1D(
        filters=64,
        activation='relu',
        strides=1,
        kernel_size=3
    ),
    tf.keras.layers.MaxPool1D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Conv1D(
        filters=64,
        activation='relu',
        strides=1,
        kernel_size=3
    ),
    tf.keras.layers.MaxPool1D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

cnn_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

cnn_model.fit(
    x=X_train,
    y=y_train,
    verbose=2,
    batch_size=128, 
    epochs=10,
    shuffle=True
)

preds = cnn_model.predict(X_test)
preds_int = []
for pred in preds.flatten():
    preds_int.append(int(np.round(pred)))

cnn_model_score = accuracy_score(y_pred=preds_int, y_true=y_test)
cnn_cfmatrix = confusion_matrix(y_pred=preds_int, y_true=y_test)

# plot all model scores
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
bars = ax1.bar(range(0, 3), [rnn_model_score, sklearn_model_score, cnn_model_score], color=['tomato', 'lightblue', 'pink'])
ax1.set_xticks(range(0, 3))
ax1.set_xticklabels(['LSTM', 'MultinomialNB', 'CNN 1D'])
for p in bars.patches:
    ax1.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), xytext=(10, 10), va='center', ha='center', textcoords='offset points')
ax1.legend()

# confusion matrix for all models
sns.heatmap(data=rnn_cfmatrix, annot=True, ax=ax2, fmt='.2f')
sns.heatmap(data=sklearn_cfmatrix, annot=True, ax=ax3, fmt='.2f')
sns.heatmap(data=cnn_cfmatrix, annot=True, ax=ax4, fmt='.2f')

ax1.set_title('Scores')
ax2.set_title('Confusion Matrix (RNN LSTM Model)')
ax3.set_title('Confusion Matrix (MultinomialNB Model)')
ax4.set_title('Confusion Matrix (CNN 1D Model)')

plt.show()
