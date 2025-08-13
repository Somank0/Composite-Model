# DRN Conda environment

To create the environment for DRN or Composite DRN :

> Copy the myenv.yml file to the user area.

> Install Conda/miniconda3

> Run the command 

```
conda env create -f myenv.yml
```
Activate the environment using 

```
conda activate ml
```

To submit slurm jobs, include the following lines in the slurm file

```
source [path to miniconda/conda environment]

conda activate ml
```

A sample slurm file is present as train.slurm.
