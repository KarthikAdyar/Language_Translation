

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import io
import os
import h5py


def conv_lower(sentence):
    sentence = sentence.lower()
    wds = sentence.split()
    new_wds = []
    for sntns in wds:
        res = [char for char in sntns if char.isalnum()]
        new_wds.append("".join(res))

    soln = ""
    for wd in new_wds:
        soln += wd + " "
    return (soln.strip())


# Assign the data path.
data_path = os.path.dirname(__file__) + '/tulu'

# Read in the data.
lines = io.open(data_path, encoding = "utf-8").read().split("\n")
lines  = lines[:-1]

# Split the data into input and target sequences.
lines = [line.split("\t") for line in lines]

# We define the starting signal to be "\t" and the
# ending signal to be "\n". These signals tell the
# model that when it sees "\t" it should start
# producing its translation and produce "\n" when
# it wants to end its translation. Let us add
# "\t" to the start and "\n" to the end
# of all input and output sentences.
#lines = [("\t" + conv_lower(line[0]) + "\n", "\t" + line[1] + "\n") for
            #line in lines]
l=[]
for line in lines:
    try:
        l.append(("\t" + conv_lower(line[0]) + "\n", "\t" + conv_lower(line[1]) + "\n"))
    except:
        continue
lines = l

input_lengths = np.array([len(line[0]) for line in lines])
output_lengths = np.array([len(line[1]) for line in lines])

english = 75
tulu = 85

line1 = []
for i in range(len(input_lengths)):
    if input_lengths[i] < english and output_lengths[i] < tulu:
        line1 = line1 + [lines[i]]

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 2869  # Number of samples to train on.

input_texts = [(line[0]) for line in line1]
target_texts = [(line[1]) for line in line1]

input_characters = set()
target_characters = set()

for input_text in input_texts:
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
for target_text in target_texts:
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index.get(char)] = 1.

#model = load_model('s2s.h5')

model = load_model(os.path.dirname(__file__) + '\s2s.h5')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def return_sentences(url):
    input_sentence = url
    input_sentence = input_sentence.lower()
    alphanumeric_sentence = ""

    for character in input_sentence:
        if character == " ":
            alphanumeric_sentence += character
        if character.isalnum():
            alphanumeric_sentence += character

    alphanumeric_sentence = "\t" + alphanumeric_sentence + "\n"
    test_sentence_tokenized = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(alphanumeric_sentence):
        test_sentence_tokenized[0, t, input_token_index.get(char)] = 1.
    print(alphanumeric_sentence)
    return decode_sequence(test_sentence_tokenized).capitalize()

