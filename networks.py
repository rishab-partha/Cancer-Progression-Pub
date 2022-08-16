import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import matplotlib.pyplot as plt
import drugs
import load_tcga
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

def plot(net_history, metric_id):
    plt.plot(net_history.history[metric_id])
    plt.plot(net_history.history['val_' + metric_id], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric_id)
    plt.legend([metric_id, 'val_'+metric_id])

def generate_dataset(fulldata, train_percent):
    inputs = np.asarray(fulldata[:, 1]).astype('str')
    inputs = tf.strings.split(inputs).numpy()
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, dtype=object, padding = 'post', value = '')
    inputs = np.asarray(inputs).astype('str')
    
    print(inputs.shape)

    labels = np.asarray(fulldata[:, 2]).astype('int')

    occurences = np.bincount(labels)
    print(occurences)
    class_weights = dict()
    for i in range(len(occurences)):
        class_weights[i] = (1 / occurences[i])*(inputs.shape[0])/2.0
    print(class_weights)

    if train_percent == 100:
        train_inputs = inputs
        test_inputs = inputs
        train_labels = labels
        test_labels = labels
    else:
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, train_size = train_percent)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

    BUFFER_SIZE = 200
    BATCH_SIZE =  16

    unshuffled_train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    unshuffled_test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, class_weights

def load_dataset_one_class_stage(classname, train_percent):
    fulldata, mutationlist, mutationnum = load_tcga.processoneclass(classname)
    fulldata, stagelist = load_tcga.processforstageoutput(fulldata)

    unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, class_weights = generate_dataset(fulldata, train_percent)

    return unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, mutationlist, mutationnum, stagelist, class_weights

def load_dataset_one_class_stage_preprocessed(classname, train_percent, ignore_stages, gene_num = 100):
    fulldata, mutationlist = load_tcga.load_one_class_from_heatmap(classname, ignore_stages=ignore_stages, gene_num=gene_num)
    fulldata, stagelist = load_tcga.processforstageoutput(fulldata)
    
    unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, class_weights = generate_dataset(fulldata, train_percent)
    return unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, mutationlist, len(mutationlist), stagelist, class_weights


def create_encoder_decoder(mutationlist):
    encoder = tf.keras.layers.experimental.preprocessing.StringLookup(num_oov_indices=0, vocabulary = mutationlist)
    decoder = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = encoder.get_vocabulary(), invert=True)
    return encoder, decoder

def create_model_stage_prediction(encoder, output_size, embedding_dim=256, rnn_units=64):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim = len(encoder.get_vocabulary()),
            output_dim = embedding_dim,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(output_size)
    ])
    return model

def generate_confusion_matrix(model, test_set, stagelist, plot_axes):
    y = np.concatenate([y for x, y in test_set], axis=0)
    Y_pred = model.predict(test_set)
    Y_pred = tf.nn.softmax(Y_pred)
    y_pred = np.argmax(Y_pred, axis = 1)
    print("preds: {}, {}".format(y_pred.dtype, y_pred.shape))
    print("y_regression: {}, {}".format(y.dtype, y.shape))

    cf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', xticklabels=stagelist, yticklabels=stagelist, ax = plot_axes)

#Taken from https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045#gistcomment-3066704
def generate_roc_curve(model, test_set, stagelist, plot_axes, average="macro"):
    y_test = np.concatenate([y for x, y in test_set], axis = 0)
    Y_pred = model.predict(test_set)
    Y_pred = tf.nn.softmax(Y_pred)
    y_pred = np.asarray(Y_pred)

    all_classes = []
    for i in range(len(stagelist)):
        all_classes.append(i)
    lb = LabelBinarizer()
    lb.fit(all_classes)
    y_test = lb.transform(y_test)

    for (idx, c_label) in enumerate(stagelist):
        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        plot_axes.plot(fpr, tpr, label = '%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    plot_axes.plot(fpr, fpr, 'b-', label = 'Random Guessing')

    return roc_auc_score(y_test, y_pred, average = average)


def run_model_stage_prediction(classname, train_percent, save_loc, ignore_stages=None, heatmap=False, epochs=10, gene_num = 100):
    if not heatmap:
        unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, mutationlist, mutationnum, stagelist, class_weights = load_dataset_one_class_stage(classname, train_percent)
    else:
        unshuffled_train_dataset, unshuffled_test_dataset, train_dataset, test_dataset, mutationlist, mutationnum, stagelist, class_weights = load_dataset_one_class_stage_preprocessed(classname, train_percent, ignore_stages, gene_num=gene_num)

    stagelist_for_roc = stagelist + ['Random Guessing']
    encoder, decoder = create_encoder_decoder(mutationlist)
    model = create_model_stage_prediction(encoder, len(stagelist))
    print(len(encoder.get_vocabulary()))
    model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    for input_example_batch, target_example_batch in train_dataset.take(1):
        print(input_example_batch.numpy())
        encoded = encoder(input_example_batch)
        print(encoded.numpy())
        print(decoder(encoded).numpy())
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions)
        print(example_batch_predictions.shape)
    print(model.summary())
    history = model.fit(train_dataset, epochs = epochs, validation_data=test_dataset, class_weight=class_weights)

    test_loss, test_acc = model.evaluate(test_dataset)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    train_loss, train_acc = model.evaluate(train_dataset)
    print('Train Loss: {}'.format(train_loss))
    print('Train Accuracy: {}'.format(train_acc))

    model.save(os.path.join("models", save_loc))

    plt.figure(figsize = (16,16))
    ax = plt.subplot(3, 2, 1)
    generate_confusion_matrix(model, unshuffled_test_dataset, stagelist, ax)
    ax.set_title("Test Confusion Matrix")
    ax = plt.subplot(3, 2, 2)
    generate_roc_curve(model, unshuffled_test_dataset, stagelist, ax)
    ax.set_title("Test ROC Curve")
    plt.legend(stagelist_for_roc)
    ax=plt.subplot(3, 2, 3)
    generate_confusion_matrix(model, unshuffled_train_dataset, stagelist, ax)
    ax.set_title("Train Confusion Matrix")
    ax = plt.subplot(3, 2, 4)
    generate_roc_curve(model, unshuffled_train_dataset, stagelist, ax)
    ax.set_title("Train ROC Curve")
    plt.legend(stagelist_for_roc)
    plt.subplot(3,2,5)
    plot(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(3,2,6)
    plot(history, 'loss')
    plt.ylim(0, None)
    plt.savefig(os.path.join("models", save_loc, "image_data"))


#load_dataset_one_class_stage_preprocessed('BRCA', 0.8, ignore_stages = {4})
# run_model_stage_prediction("BRCA", 0.8, "stage_model_0.8Train")
