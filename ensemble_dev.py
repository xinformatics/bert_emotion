### Maximum voting over different classifiers
### Once we have trained different models-we can take ensemble

import pandas as pd
import numpy as np




# final_predictions now contains the ensemble's predictions
model_0 = pd.read_csv('model_0.csv')
model_1 = pd.read_csv('model_1.csv')
model_2 = pd.read_csv('model_2.csv')
model_3 = pd.read_csv('model_3.csv')
model_4 = pd.read_csv('model_4.csv')
model_5 = pd.read_csv('model_5.csv')
model_6 = pd.read_csv('model_6.csv')
model_7 = pd.read_csv('model_7.csv')
model_8 = pd.read_csv('model_8.csv')
model_9 = pd.read_csv('model_9.csv')


import pandas as pd
import numpy as np
from scipy.stats import mode

# Extract relevant columns and convert to numpy arrays
models_predictions = [model_0.iloc[:, 1:].to_numpy(), 
                      model_1.iloc[:, 1:].to_numpy(),
                      model_2.iloc[:, 1:].to_numpy(),
                      model_3.iloc[:, 1:].to_numpy(),
                      model_4.iloc[:, 1:].to_numpy(),
                      model_5.iloc[:, 1:].to_numpy(),
                      model_6.iloc[:, 1:].to_numpy(),
                      model_7.iloc[:, 1:].to_numpy(),
                      model_8.iloc[:, 1:].to_numpy(),
                      model_9.iloc[:, 1:].to_numpy()]

# Perform maximum voting
# Stack the predictions for easier computation
stacked_predictions = np.stack(models_predictions, axis=-1)

# Use mode for maximum voting (axis=-1 means voting across models for each sample)
voting_results = mode(stacked_predictions, axis=-1)[0].squeeze()

predict_df = model_0.copy()
predict_df.iloc[:, 1:] = voting_results

# Save the result to a new CSV file
predict_df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name=f'submission.csv'))


### then submit this to condabench
