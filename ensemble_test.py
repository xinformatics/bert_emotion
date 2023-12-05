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

#clear session function - useful later
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

######### load test data-- use the same code as nn.py; just replace dev by test for final prediction
df_dev = pd.read_csv('test-in_clean.csv') ## this is a clean text csv; procedure in cleaning.py

#
data_dev = []
# Iterate over each row in the dataframe
for _, row in df_dev.iterrows():
    text = row[0]
    labels = []
    # print(text)
    # Assuming the presence of a label is indicated by 1
    for i in range(1, 8):  # For the next seven columns
        # print(row[i])
        if row[i] == 1:
            labels.append(1)
        else:
            labels.append(0)
    # print(labels)
    data_dev.append((text, labels))

dev_texts, dev_labels = zip(*data_dev)

texts_dev =  [text for text in dev_texts]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = TFBertModel.from_pretrained("bert-base-uncased")


#convert test text to tokens
test_input_ids=[]
test_attention_masks=[]

for sent in texts_dev:
    bert_inp_test=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =35,pad_to_max_length = True,return_attention_mask = True, truncation=True)
    test_input_ids.append(bert_inp_test['input_ids'])
    test_attention_masks.append(bert_inp_test['attention_mask'])

test_input_ids=np.asarray(test_input_ids)
test_attention_masks=np.array(test_attention_masks)


#Load previous trained BERT models (finetuning with Global Average Pooling)
model0 = tf.keras.models.load_model('model0_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_0 = np.where(model0.predict([test_input_ids,test_attention_masks]) >= 0.41, 1, 0)
clear_sess()

model1 = tf.keras.models.load_model('model1_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_1 = np.where(model1.predict([test_input_ids,test_attention_masks]) >= 0.40, 1, 0)
clear_sess()

model2 = tf.keras.models.load_model('model2_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_2 = np.where(model2.predict([test_input_ids,test_attention_masks]) >= 0.37, 1, 0)
clear_sess()

model3 = tf.keras.models.load_model('model3_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_3 = np.where(model3.predict([test_input_ids,test_attention_masks]) >= 0.38, 1, 0)
clear_sess()

model4 = tf.keras.models.load_model('model4_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_4 = np.where(model4.predict([test_input_ids,test_attention_masks]) >= 0.41, 1, 0)
clear_sess()

model5 = tf.keras.models.load_model('model5_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_5 = np.where(model5.predict([test_input_ids,test_attention_masks]) >= 0.42, 1, 0)
clear_sess()

model6 = tf.keras.models.load_model('model6_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_6 = np.where(model6.predict([test_input_ids,test_attention_masks]) >= 0.40, 1, 0)
clear_sess()

model7 = tf.keras.models.load_model('model7_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_7 = np.where(model7.predict([test_input_ids,test_attention_masks]) >= 0.43, 1, 0)
clear_sess()

model8 = tf.keras.models.load_model('model8_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_8 = np.where(model8.predict([test_input_ids,test_attention_masks]) >= 0.41, 1, 0)
clear_sess()

model9 = tf.keras.models.load_model('model9_test_bert', custom_objects={"TFBertModel": transformers.TFBertModel}) #BERT
test_preds_9 = np.where(model9.predict([test_input_ids,test_attention_masks]) >= 0.41, 1, 0)
clear_sess()

models_predictions = [test_preds_0,test_preds_1,test_preds_2,test_preds_3,test_preds_4
                     test_preds_5,test_preds_6,test_preds_7,test_preds_8,test_preds_9]

stacked_predictions = np.stack(models_predictions, axis=-1)


# Use mode for maximum voting (axis=-1 means voting across models for each sample)
voting_results = mode(stacked_predictions, axis=-1)[0].squeeze()

predict_df = df_dev.copy()
predict_df.iloc[:, 1:] = voting_results

# Save the result to a new CSV file
predict_df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name=f'submission.csv'))