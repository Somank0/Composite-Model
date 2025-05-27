## CompositeDRN

This is a composite network that uses a DRN and a classifier. 

The output from the DRN and the true/fake value of the target are inputs to the classifier. The classifier, based on the DRN output, should predict if the target value is true or fake.

The models are written in Composite_Models.py

To train the model :

```
/[path to this folder]/train --data_folder [folder with pickle files] --drn_input_dim [number of input features] --num_epochs [number of epochs] --batch_size 40 --learning_rate 0.0001
``` 

For inference :

```
/[path to this folder]/train --data_folder [folder with pickle files] --drn_input_dim [number of input features] --batch_size 40 --predict_only
```

