from multistep_dataset import MultistepDataset
import math
import tool_box
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from keras.utils import set_random_seed
set_random_seed(1234)


class HyperModel:

    def __init__(self, model_name, dir_name=f"{os.getcwd()}/hp_results"):
        self.model_name = f"{model_name}"
        self._dir_name = dir_name
        self.path = f"{os.getcwd()}/trained_hyperparameter_models/{self.model_name}.keras"


    def load_model(self):
        if os.path.exists(self.path) == False:
            print(tool_box.color_string("red", f"\n\n\tNO MODEL FOUND IN PATH: {self.path};\n\tcall method 'run_grid_search' to generate new model\n"))
            return None
        else:
            return keras.models.load_model(self.path)
        

    def load_model_params(self):
        '''
            returns best params found during latest grid search; returns object to be used by hypermodel build function
            or returned as dict with load_best_params (.value key used)
        '''
        if os.path.exists(self.path) == False:
            return None
        else:
            model = keras.models.load_model(self.path)
            input_shape = model.layers[0].input.shape
            self.n_features = input_shape[2]
            self.n_steps_in = input_shape[1]
            self.n_steps_out = model.layers[-1].output.shape[1]

            self.tuner = kt.GridSearch(
                hypermodel=self._build_fn,
                objective='val_loss',
                directory=self._dir_name,
                project_name=self.model_name,
              
            )

            self.tuner.reload()
            best_params = self.tuner.get_best_hyperparameters()[0]
            return best_params

    def _check_epoch(self, epoch, logs, hyperparams_path=None):
        print(tool_box.color_string('green', f"\n{self.model_name}.....\n"))
        val_loss = logs["val_loss"]
        if hyperparams_path == None:
            params_path = f"{os.getcwd()}/hyperparameters/{self.model_name}_best_params.pkl"
        else:
            params_path = hyperparams_path
    

        #check if epoch is greater than current max and if the loss is less than .5 X the current best loss
        epoch_check = epoch >= self.current_max_epoch * 1.5
        loss_check = val_loss <= self.best_loss + 0.05
        if epoch > self.current_max_epoch and epoch_check and loss_check:

            self.current_max_epoch = epoch
            self.best_loss = val_loss


            #save best
            optimizer = self.model.optimizer
            learning_rate = optimizer.get_config()["learning_rate"]
            units = self.model.layers[0].get_config()["units"]
            loss = self.model.loss
            l1_reg = self.model.layers[2].get_config()["kernel_regularizer"]["config"]["l1"]
            l2_reg = self.model.layers[2].get_config()["kernel_regularizer"]["config"]["l2"]
            #create legible config details to display
            params_text = {
                "l1": l1_reg,
                "l2": l2_reg,
                "units": units,
                "learning_rate": learning_rate,
                "optimizer": optimizer.get_config()["name"],
                "loss_function": loss,
                "epochs": self.current_max_epoch,
                "val_loss": self.best_loss
            }
            
           # self.losses.append(self.best_loss)
            self.best_param_details["compile_configs"] = self.current_params_json
            self.best_param_details["config_text"] = params_text
            self.best_param_details["epochs"] = self.current_max_epoch
    
        
            tool_box.Create_Pkl(params_path, self.best_param_details)



    def display_best_params(self, logs):
      
        self.losses.append(logs["val_loss"])
        print(tool_box.color_string("green",(f"\n\tBEST MODEL PARAMS:\n")))
        for k, v in self.best_param_details["config_text"].items():
            print(tool_box.color_string("yellow", k),":", v)
        print("\n")
   
        

                
    def run_grid_search(self, df, n_steps_in, n_steps_out, n_stride=1, max_trials=50,hyperparams_path=None):
        
        
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out


        #load dataset
        dataset = MultistepDataset(df, steps_in=n_steps_in,steps_out=n_steps_out, stride=n_stride)
        
        self.n_features = dataset.feature_count
     
        # #define tuner
        self.tuner = kt.GridSearch(
            hypermodel=self._build_fn,
            overwrite=True,
            objective='val_loss',
            directory=self._dir_name,
            project_name=self.model_name,
            max_trials=max_trials,
              
        )

        #set path to save configurations in _check_epoch callback if epoch performance tops current best (self._load_configs uses the data to recompile model after training)
        current_best_path = f"{os.getcwd()}/hyperparameters/{self.model_name}_best_params.pkl"

        #if best_params already exist, load them - else create new dict to start training with
        if os.path.exists(current_best_path):
            #load_config if exists; self.model, self.best_params, self.config_text, self.config_details,  and self.best_epochs set in method
            self.load_from_config()

            #load data dict created in _check_epoch callback that contains model configurations and display information
            current_best = tool_box.Load_Pkl(current_best_path)
            self.best_loss = current_best["config_text"]["val_loss"]
            self.current_max_epoch = current_best["epochs"]

            self.best_params = self.model.to_json()
            self.best_param_details = {
                "compile_configs": self.best_params,
                "config_text": current_best["config_text"],
                "epochs": self.current_max_epoch
            }
            self.losses = []
        else:
            #if no configs exist (grid_search being run for first time for model), set initial training params
            self.current_max_epoch = 1
            self.best_loss = math.inf
            self.best_params = self.current_params_json
            self.losses = []
            self.best_param_details = {}

        #set checkpoint path to pass to ModelCheckpoint callback - saves model on best "val_loss" (or whatever is passed to as method's monitor argument)
        checkpoint_path = f"{os.getcwd()}/model_checkpoints/{self.model_name}_checkpoint.keras"
        #set callbacks
        cb = [
              keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self._check_epoch(epoch, logs,hyperparams_path=hyperparams_path), on_train_end=lambda logs: self.display_best_params(logs)),
              keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.05, patience=5, start_from_epoch=2, restore_best_weights=True),
              keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss",save_best_only=True),
         ]
        
        #initialize tuner search using scaled data; set epochs as high as possible - early stopping will limit runs
        self.tuner.search(dataset.scaled_X_train, dataset.scaled_y_train,validation_split=0.2 , epochs=5000, callbacks=cb)

        #after search is complete, return best model found during search by tuner
        best_model = self.tuner.get_best_models(num_models=1)[0]
        best_model.summary()

        #save best_model to f"{os.getcwd()}/trained_hyperparameter_models/{self.model_name}.keras"
        best_model.save(self.path)
        print("DONE SEARCHING...")
     
        
    
    def _build_fn(self,hp):

        #instantiate sequential model
        self.model = keras.Sequential()
    
    
        #l1 reg vals to pass
        hp_l1_l2_reg = hp.Choice('l1_l2', values=[0.0, 0.01, 0.001, 0.0001])
        #units to test
        hp_units = hp.Choice("units", values=[10, 20, 30])

      

        #for layers in range(hp.Int("layer_count", 1, 10)):
        self.model.add(keras.layers.LSTM(units=hp_units, activation="relu", input_shape=(self.n_steps_in, self.n_features)))
        self.model.add(keras.layers.RepeatVector(self.n_steps_out))
        self.model.add(keras.layers.LSTM(units=hp_units, activation="relu", return_sequences=True, input_shape=(self.n_steps_in, self.n_features), kernel_regularizer=keras.regularizers.l1_l2(hp_l1_l2_reg)))

        self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.n_features)))

    


        #add optimizer and losses
        hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
        hp_optimizer = hp.Choice("optimizer", values=["adam", "nadam"])
        hp_losses = hp.Choice("loss", values=["huber"])

        #compile model
        if hp_optimizer == "adam":
            self.model.compile(loss=hp_losses, optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate))
        else:
            self.model.compile(loss=hp_losses, optimizer=keras.optimizers.Nadam(learning_rate=hp_learning_rate))

        #create temp pointer to reference model params in training callback

        self.current_params_json = self.model.to_json()
        
        return self.model


    def best_param_predictions(self, symbol, n_steps_in, n_steps_out, target_date=None,future_forecast=False):
     
        if target_date != None and future_forecast != False:
            print(tool_box.color_string("red", f"\n\ntWARNING: YOU ARE PASSING target_date AS PAST DATE AND future_forecast AS TRUE"))
            return None
           
        else:
            configs_path = f"{os.getcwd()}/hyperparameters/{self.model_name}_best_params.pkl"
            #check if configs created by run_grid_search exist; if not, return none and exit (need to run grid_search_first)
            if os.path.exists(configs_path) == False:
                print(tool_box.color_string('red',f"\n\n\tNO MODEL CONFIGS FOUND AT  PATH: {configs_path}\nUSE 'run_grid_search' to generate new configs and run prediction again\n"))
                return None
            
            else:
                print(tool_box.color_string('green',f"\n\n\tLOADING MODEL FROM CONFIG"))
                #set model in load_from_config function
                self.load_from_config(configs_path)
                #compile self.model with config text params
                optimizers = {"Nadam": keras.optimizers.Nadam, "Adam": keras.optimizers.Adam}
                learning_rate = self.config_text["learning_rate"]
                optimizer = optimizers[self.config_text["optimizer"]]
                loss = self.config_text["loss_function"]
                self.best_params = self.model.to_json()
                self.model.compile(loss=loss, optimizer=optimizer(learning_rate=learning_rate))
         

                #load dataset
                dataset = MultistepDataset(symbol, n_steps_in, n_steps_out)

                if type(dataset) != MultistepDataset:
                    print(tool_box.color_string("red", f"ERROR: CHECK MULTISTEP STATUS RETURNED AS FALSE\n"))
                    return None
                elif dataset.status == False:
                    print(tool_box.color_string("red", f"ERROR: CHECK MULTISTEP STATUS RETURNED AS FALSE\n"))
                    return None
                else:
                    self.model.summary()
                    
                    #set training params
                    self.best_epochs = 100000
                    cb = [keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.05, patience=5, start_from_epoch=2, restore_best_weights=True)]

                    #train model with scaled data
                    self.model.fit(dataset.scaled_X_train, dataset.scaled_y_train, validation_data=(dataset.scaled_X_val, dataset.scaled_y_val), epochs=self.best_epochs, callbacks=cb)

                    #use training model to make prediction with scaled X_test
                    pred = self.model.predict(dataset.scaled_X_test).reshape(-1, dataset.feature_count)

                    #create new df that contains output of prediction + actual prices
                    
                    #set pred_df columns to same as dataset.prices and index set to dataset.X_test_indices
                    pred_df = pd.DataFrame(dataset.scaler.inverse_transform(pred), columns=[f'predicted_{c}' for c in dataset.prices.columns],index=dataset.X_test_indices)
                    
                    #remove ratio columns and keep only price related cols:  predicted_open  predicted_high  predicted_low  predicted_close  predicted_adjClose  predicted_volume  predicted_unadjustedVolume  predicted_change  predicted_changePercent  predicted_vwap
                    pred_df = pred_df[[f"predicted_{c}" for c in dataset.price_cols if f"predicted_{c}" in pred_df.columns]]
                    
                    #subset original prices (original prices with no altered/removed data) using pred_df index and columns
                    prices_df = dataset.original_prices.loc[pred_df.index,dataset.prices.columns]
                    prices_df = prices_df.rename(columns={c: f'actual_{c}' for c in prices_df.columns})

                    #remove ratio columns from actual price_data (same as pred_df)
                    prices_df = prices_df[[f"actual_{c}" for c in dataset.price_cols if f"actual_{c}" in prices_df.columns]]

                    #prepare both dataframes for combining by droping columns
                    prices_df = prices_df.reset_index().drop(['index'], axis=1)
                    pred_df = pred_df.reset_index().drop(['index'], axis=1)
                   
                    #combine Dataframes
                    df = pd.concat([pred_df, prices_df], axis=1)
                   
                    #create alternating actual/predicted columns so related data are presented next to eachother (ex: actual_prices, predicted_prices)
                    alternating_cols = np.vstack((pred_df.columns,prices_df.columns)).ravel('F')     
                    df = df[alternating_cols]

                    #set combined_df index to X_test indices (target output dates)
                    df.index = dataset.X_test_indices
                    return df
    

    def load_from_config(self,configs_path=None):
        '''
            key = config you want maximized/minimized (val_loss will return lowest val_loss and epochs will return max epochs)
        '''
        if configs_path == None:
            configs_path = f"{os.getcwd()}/hyperparameters/{self.model_name}_best_params.pkl"
        else:
            configs_path = configs_path
        
        #load config dict from file
        self.config_details = tool_box.Load_Pkl(configs_path)
        #set compile configs
        self.compile_configs = self.config_details["compile_configs"]
        #set config_text (used to display status of best model params)
        self.config_text = self.config_details["config_text"]
        #set epochs (**currently not in use due to early stop in both grid_search and predict methods - remove hardcoded value in those methods to enable)
        self.best_epochs = self.config_details["epochs"]
        #compile model
        self.model = keras.models.model_from_json(self.compile_configs)
        #print model params
        print(tool_box.color_string("green",(f"\n\tBEST MODEL PARAMS:\n")))
        for k, v in self.config_text.items():
            print(tool_box.color_string("yellow", k),":", v)
        print("\n")
      


original_df = tool_box.Load_Pkl("aapl.pkl")

#print(original_df.columns)

steps_in = 5
steps_out = 5
stride = 1


#multistep_dataset = MultistepDataset(original_df, steps_in, steps_out, stride)


model = HyperModel("test")
model.run_grid_search(df=original_df,
                      n_steps_in=steps_in,
                      n_steps_out=steps_out,
                      n_stride=stride)



#  ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 5, 11), found shape=(None, 11)

