import tool_box
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sys

class MultistepDataset:

    def __init__(self, dataframe, steps_in, steps_out, stride):

        self.original_df = dataframe
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.stride = stride

        
    
        self._load_multistep_data()
    
    

    def display_data(self):
            
            #========== X_train ==========
            print(f"\nORIGINAL X_TRAIN (head):\n")
            print(pd.DataFrame(self.X_train.reshape(self.scaled_X_train.shape)).head())
            print(f"\nSCALED_X_TRAIN (head):\n")
            print(pd.DataFrame(self.scaled_X_train).head())
            print(f"\nREVERTED SCALED_X_TRAIN (head):\n")
            reverted_X_train = self.scaler.inverse_transform(self.scaled_X_train)
            print(pd.DataFrame(reverted_X_train).head())

            #=========== y_train ==================
            print(f"\nORIGINAL y_TRAIN (head):\n")
            print(pd.DataFrame(self.y_train.reshape(self.scaled_y_train.shape)).head())
            print(f"\nSCALED_y_TRAIN (head):\n")
            print(pd.DataFrame(self.scaled_y_train).head())
            print(f"\nREVERTED SCALED_y_TRAIN (head):\n")
            reverted_y_train = self.scaler.inverse_transform(self.scaled_y_train)
            print(pd.DataFrame(reverted_y_train).head())


            #========== X_test ========================
            print(f"\nORIGINAL X_TEST (head):\n")
            print(pd.DataFrame(self.X_test.reshape(self.scaled_X_test.shape)).head())
            print(f"\nSCALED_X_TEST (head):\n")
            print(pd.DataFrame(self.scaled_X_test).head())
            print(f"\nREVERTED SCALED_X_TEST (head):\n")
            reverted_X_test = self.scaler.inverse_transform(self.scaled_X_test)
            print(pd.DataFrame(reverted_X_test).head())    

            #========== y_test ========================
            print(f"\nORIGINAL y_TEST (head):\n")
            print(pd.DataFrame(self.y_test.reshape(self.scaled_y_test.shape)).head())
            print(f"\nSCALED_y_TEST (head):\n")
            print(pd.DataFrame(self.scaled_y_test).head())
            print(f"\nREVERTED SCALED_y_TEST (head):\n")
            reverted_y_test = self.scaler.inverse_transform(self.scaled_y_test)
            print(pd.DataFrame(reverted_y_test).head())

            #============ DISPLAY LAST INPUT/OUTPUT INDICES IN TRAIN/TEST =======
            print(tool_box.color_string('green', f"\n\tLAST INPUT INDEX OF X_TRAIN DATA ({self.steps_in} steps_in): \n"),  f"{self.X_train_df_windows[-1]}")
            print(tool_box.color_string('green', f"\n\tLAST OUTPUT INDEX OF y_TRAIN DATA  ({self.steps_out} steps_out): \n"), f"{self.y_train_df_windows[-1]}\n")
            print(tool_box.color_string('red', f"\n\tLAST INPUT INDEX OF X_TEST DATA: ({self.steps_in} steps_in): \n"), f"{self.X_test_df_windows[-1]}")
            print(tool_box.color_string('red', f"\n\tLAST OUTPUT INDEX OF Y_TEST DATA:  ({self.steps_out} steps_out): \n"), f"{self.y_test_df_windows[-1]}\n\n")

    def _load_multistep_data(self):
    
        if type(self.original_df) != pd.DataFrame:
            print(tool_box.color_string('red', f'\nINVALID DATATYPE FOUND FOR SELF.ORIGINAL_DF; PASS DATAFRAME\n'))
            return None
        else:
                    
        
            #save reference to original data columns and indices
            self.original_index = list(self.original_df.index)[::-1]
            self.original_columns = self.original_df.columns

            self.numerical_columns = self.original_df.select_dtypes("number").columns
            self.categorical_columns = [c for c in self.original_columns if c not in self.numerical_columns]
            self.feature_count = len(self.numerical_columns)

            #create input windows with stride
            input_windows = list(np.lib.stride_tricks.sliding_window_view(self.original_index, self.steps_in, axis=0)[::self.stride])

            #for each window in the input windows, find index in original index and return the next steps out indices
            self.X_train_indices = []
            self.y_train_indices = []


            for window in input_windows:
                last_index = sorted(window)[-1]
                try:  
                    next_window_start_index = self.original_index.index(last_index) + 1
                    next_window_end_index = next_window_start_index + self.steps_out
                    output_window = self.original_index[next_window_start_index: next_window_end_index]
                    if len(output_window) < self.steps_out:
                        #set test_indices for in and out here by using last appended input in input_windows (in) and output_windows (out) - break
                        break
                    else:
                        self.y_train_indices.append(output_window)
                        self.X_train_indices.append(window)
                       
                except Exception as e:
                    print(tool_box.color_string('red',f"ERROR PARSING INDICES: {e}\n\nSETTING MULTISTEP_DATASET STATUS TO FALSE\n"))
                    self.status = False

            last_input_window = self.X_train_indices.pop()
            last_output_window = self.y_train_indices.pop()

            self.X_test_indices = [last_input_window]
            self.y_test_indices = [last_output_window]

            #====== use indices to create train/test data  ========================

            self.X_train_df_windows = [self.original_df.loc[window][self.numerical_columns] for window in self.X_train_indices] #list of dataframes for each input
            self.y_train_df_windows = [self.original_df.loc[window][self.numerical_columns] for window in self.y_train_indices]

            self.X_test_df_windows =  [self.original_df.loc[window][self.numerical_columns] for window in self.X_test_indices]
            self.y_test_df_windows =  [self.original_df.loc[window][self.numerical_columns] for window in self.y_test_indices]

            #Create train/test and reshape windows for LSTM model input
            X_train = np.array([df.values for df in self.X_train_df_windows])
            y_train = np.array([df.values for df in self.y_train_df_windows])

            X_test = np.array([df.values for df in self.X_test_df_windows])
            y_test = np.array([df.values for df in self.y_test_df_windows])

            self.X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.feature_count))
            self.y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], self.feature_count))

            self.X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.feature_count))
            self.y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], self.feature_count))

            #save original shape to restore later
            self.original_shape = (-1, self.X_train.shape[1], self.X_train.shape[2])

            #scale data
            numerical_data = self.original_df.select_dtypes("number").values
            self.scaler = StandardScaler()
            self.scaler.fit(numerical_data)

            self.scaled_X_train = self.scaler.transform(self.X_train.reshape(-1, self.feature_count))
            self.scaled_y_train = self.scaler.transform(self.y_train.reshape(-1, self.feature_count))

            self.scaled_X_test = self.scaler.transform(self.X_test.reshape(-1, self.feature_count))
            self.scaled_y_test = self.scaler.transform(self.y_test.reshape(-1, self.feature_count))

            #set status to true
            self.status = True
           # print(self.scaled_X_train)
            #print(self.X_train)

            # else:
            #     self.X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.feature_count))
            #     self.y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], self.feature_count))

            #     self.X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], self.feature_count))
            #     self.y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], self.feature_count))
            #     #scale data
               
            #     #save original shape to restore later
            #     self.original_shape = (-1, self.X_train.shape[1], self.X_train.shape[2])
            #     self.X_train = self.X_train.reshape(-1, self.feature_count)
            #     self.X_test = self.X_test.reshape(-1, self.feature_count)
            #     self.X_val = self.X_val.reshape(-1, self.feature_count)
            #     self.y_train = self.y_train.reshape(-1, self.feature_count)
            #     self.y_val = self.y_val.reshape(-1, self.feature_count)
            #     self.y_test = self.y_test.reshape(-1, self.feature_count)

            #     self.scaler = StandardScaler()
            #     self.scaler.fit(self.data.values)

            #     self.scaled_X_train = self.scaler.transform(self.X_train).reshape(self.original_shape)
            #     self.scaled_X_test = self.scaler.transform(self.X_test).reshape(self.original_shape)
            #     self.scaled_X_val = self.scaler.transform(self.X_val).reshape(self.original_shape)
            #     self.scaled_y_train = self.scaler.transform(self.y_train).reshape(self.original_shape)
            #     self.scaled_y_val = self.scaler.transform(self.y_val).reshape(self.original_shape)
            #     self.scaled_y_test = self.scaler.transform(self.y_test).reshape(self.original_shape)

            #     self.status = True
                
                

original_df = tool_box.Load_Pkl("aapl.pkl")
steps_in = 4
steps_out = 2
stride = 1


multistep_dataset = MultistepDataset(original_df, steps_in, steps_out, stride)

#multistep_dataset.display_data()