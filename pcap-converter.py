import csv
from datetime import datetime
import pandas as pd
import numpy as np
    
df = pd.read_csv('caida-raw.csv')

df = df.truncate(after=14240)

data_split = [0.7, 0.1, 0.2]
type_map = {'train':0, 'val':1, 'test':2}

train_num = int(len(df)*data_split[0])
test_num = int(len(df)*data_split[2])
val_num = len(df) - train_num - test_num 

print ("train_num, val_num, test_num: ", train_num, val_num, test_num)

# Select categorical columns
categorical_columns = df.select_dtypes(include=['object'])

# Convert categorical columns to NumPy array
categorical_array = categorical_columns.to_numpy(dtype=str)
print(categorical_array.shape)

# Save the array as a NumPy file
np.save('data/pcap/X_cat_train.npy', categorical_array[:train_num])
np.save('data/pcap/X_cat_val.npy', categorical_array[train_num:train_num+val_num])
np.save('data/pcap/X_cat_test.npy', categorical_array[train_num+val_num:train_num+val_num+test_num])

# Select categorical columns
numeric_columns = df.select_dtypes(include=['int64'])

# Convert categorical columns to NumPy array
numeric_array = numeric_columns.to_numpy()

# Save the array as a NumPy file
np.save('data/pcap/X_num_train.npy', numeric_array[:train_num])
np.save('data/pcap/X_num_val.npy', numeric_array[train_num:train_num+val_num])
np.save('data/pcap/X_num_test.npy', numeric_array[train_num+val_num:train_num+val_num+test_num])

y_col = df["pkt_len"]
np.save('data/pcap/y_train.npy', y_col[:train_num])
np.save('data/pcap/y_val.npy', y_col[train_num:train_num+val_num])
np.save('data/pcap/y_test.npy', y_col[train_num+val_num:train_num+val_num+test_num])