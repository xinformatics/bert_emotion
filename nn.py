import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
import transformers
from transformers import BertTokenizer, BertModel, TFBertModel
import re
import emoji
from sklearn.metrics import f1_score, fbeta_score, classification_report, multilabel_confusion_matrix

### data loader - data cleaned in cleaning.py
df_train = pd.read_csv('train_clean.csv') #original train (25196, 8)
df_dev   = pd.read_csv('dev_clean.csv')   #original dev (3149, 8)


# Initialize an empty list to store the data
data_train = []
data_dev = []

# Iterate over each row in the dataframe and put data to list
for _, row in df_train.iterrows():
    text = row[0]
    labels = []
    # print(text)
    for i in range(1, 8):  # For the next seven columns
        # print(row[i])
        if row[i] == 1:
            labels.append(1)
        else:
            labels.append(0)
    # print(labels)
    data_train.append((text, labels))

# Iterate over each row in the dataframe
for _, row in df_dev.iterrows():
    text = row[0]
    labels = []
    # print(text)
    for i in range(1, 8):  # For the next seven columns
        # print(row[i])
        if row[i] == 1:
            labels.append(1)
        else:
            labels.append(0)
    # print(labels)
    data_dev.append((text, labels))
    
print(len(data_train), len(data_dev)) #verify if everything is correct

### zip the text and label
train_texts, train_labels = zip(*data_train)
dev_texts, dev_labels = zip(*data_dev)

# Convert texts to list of words
texts_train = [text for text in train_texts]
texts_dev =  [text for text in dev_texts]

train_labels = np.array(train_labels).astype('float32')
test_labels = dev_labels = np.array(dev_labels).astype('float32')

#BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = TFBertModel.from_pretrained("bert-base-uncased")

#Convert Text to tokens
train_input_ids=[]
train_attention_masks=[]

for sent in texts_train:
    bert_inp_train=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =35,pad_to_max_length = True,return_attention_mask = True, truncation=True)
    train_input_ids.append(bert_inp_train['input_ids'])
    train_attention_masks.append(bert_inp_train['attention_mask'])

train_input_ids=np.asarray(train_input_ids)
train_attention_masks=np.array(train_attention_masks)

test_input_ids=[]
test_attention_masks=[]

for sent in texts_dev:
    bert_inp_test=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =35,pad_to_max_length = True,return_attention_mask = True, truncation=True)
    test_input_ids.append(bert_inp_test['input_ids'])
    test_attention_masks.append(bert_inp_test['attention_mask'])

test_input_ids=np.asarray(test_input_ids)
test_attention_masks=np.array(test_attention_masks)

#clear session function
def clear_sess():
    try:
        del model
        del history
    except:
        pass
    # from tf.keras import backend as K
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    return None

#BERT finetuning with Global Average Pooling
clear_sess()
max_length =35
# Customize BERT model for multi-label classification
input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
# bert_output = bertmodel(input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
bert_output = bertmodel(input_ids, attention_mask=attention_mask).last_hidden_state

# BERT model is trainable
for layer in bertmodel.layers:
    layer.trainable = True
# Use GlobalAveragePooling1D to pool the last hidden states
pooled_output = GlobalAveragePooling1D()(bert_output)
dense_output = Dense(7, activation='sigmoid')(pooled_output)

model = Model(inputs=[input_ids, attention_mask], outputs=dense_output)

model.compile(optimizer=tf.keras.optimizers.Adam(0.000008), loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.37)])

reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', factor=0.9, patience=2, min_lr=1e-7)

#call model fit -- required
model.fit([train_input_ids,train_attention_masks],np.array(train_labels),batch_size=64,epochs=10,validation_data=([test_input_ids,test_attention_masks],np.array(test_labels)),
          callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='model5_test_bert',monitor="val_f1_score",mode="max", save_best_only=True), reduce_lr])



##once trained load the best model
model5 = tf.keras.models.load_model('model5_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT


test_preds = model5.predict([test_input_ids,test_attention_masks])

### once predicted on; vary the thresold to maximize the f1 score
for i in np.arange(0.30,0.5,0.01):
    tmp_pred = np.where(test_preds >= round(i,3), 1, 0)
    tmp_f1m = f1_score(test_labels, tmp_pred, average='micro')
    print(round(i,3), round(tmp_f1m,4))
    
#### once we have the best test_preds- we can save these models for ensembling; by changing their store location

predict_df = df_dev
predict_df.iloc[:, 1:] = test_preds

### and save predictions
predict_df.to_csv("model_5.csv", index=False)