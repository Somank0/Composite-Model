## CompositeDRN for $a$ &rarr; $\gamma$ $\gamma$

This is a composite network that uses a DRN and a classifier. 

This model was used for reconstructing the mass of pseudoscalar (_a_) in _a_ &rarr; $\gamma$ $\gamma$ samples.

The output from the DRN and the true/fake value of the target are inputs to the classifier. The classifier, based on the DRN output, should predict if the target value is true or fake.

### Architecture and Training of the Composite Network

The models are written in ***Composite_Models.py***

The Trainer is written in ***Train_Composite_Network.py***

The training wrapper is present in **_train_** which calls the functions from ***Train_Composite_Network.py***.

The data is loaded in a custom data loader class created in ***DataHandler.py***.

To train the model :

```
/[path to this folder]/train --data_folder [folder with pickle files] --drn_input_dim [number of input features] drn_output_dim [number of output features] --num_epochs [number of epochs] --batch_size 40 --learning_rate 0.0001
``` 

For inference :

```
/[path to this folder]/train --data_folder [folder with pickle files] --drn_input_dim [number of input features] --batch_size 40 --predict_only

```

The output of the Composite network could be an _n-dimensional_ vector for each event (depending on the output_dim).

The idea of the Composite network is to create the _n_ most useful features needed for mass resconsturction of _a_.  

## DNN regression 

The DNN can be used to regress the mass from the output of the Composite DRN.

### Pickles for DNN regression

> In the pickle maker, run the ***convert_pickle_to_tensor.py*** to convert the output of the Composite DRN into a pytorch tensor and create a trainining - validation split.

### Training DNN

The data is loaded using a custom dataloader class created in ***DNN_datahandler.py***.

To train the model :

```
/[path to this folder]/trainDNN --data_folder [folder with pickle files] --input_dim [number of input features] --num_epochs [number of epochs] --batch_size 40 --learning_rate 0.0001

```

For inference :

```
/[path to this folder]/trainDNN --data_folder [folder with pickle files] --input_dim [number of input features] --num_epochs [number of epochs] --batch_size 40 --learning_rate 0.0001 --predict_only 

```

It is advisable to write the logs from the training and prediction into a .log file.

The training process automatically creates a summary.npz file that stores the training and validation losses at each epoch.
