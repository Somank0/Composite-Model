import pickle 
import torch
import numpy as np
import awkward as ak
from torch_geometric.data import Data

with open('predV_64dim.pickle', 'rb') as f:
    jagged_array = pickle.load(f)#[:10000]
with open('trueclass_target.pickle','rb') as f:
    true_class=pickle.load(f)#[:10000]

jagged_array = jagged_array[true_class==1]
print("jagged_array",len(jagged_array))
with open('true_mass.pickle','rb') as f:
    trueM = pickle.load(f)#[:10000]
    trueM = np.array(trueM).flatten()
trueM_target=trueM[true_class==1]
print("trueM",len(trueM_target))
print("Filtered true from fake")

with open('trueM_target.pickle','wb') as f:
    pickle.dump(trueM_target,f)
print("Dumped Target mass")



#tensor_list = [Data(x=torch.from_numpy(ak.to_numpy(Pho).astype(np.float32).unsqueeze(0))) for Pho in jagged_array]

tensor_list = [
            Data(x=torch.from_numpy(ak.to_numpy(Pho).astype(np.float32)).unsqueeze(0))
                for Pho in jagged_array
                ]
print("tensor_list",tensor_list)
with open("cartfeat.pickle","wb") as f:
    torch.save(tensor_list,f, pickle_protocol=4)
print("Created features")

with open("trueM_target.pickle" , "rb") as f:
     trueE = pickle.load(f)
length = len(trueE)

# create train/test split
split = 0.8
train_idx = np.random.choice(length, int(split * length + 0.5), replace=False)

mask = np.ones(length, dtype=bool)
mask[train_idx] = False
valid_idx = mask.nonzero()[0]

with open("new_valididx.pickle" , "wb") as f:
    pickle.dump(valid_idx, f)

with open("new_trainidx.pickle" , "wb") as f:
    pickle.dump(train_idx, f)

