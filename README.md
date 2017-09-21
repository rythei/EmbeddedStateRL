# EmbeddedStateRL

Code implementing a latent state transition model:

S --> Z --> Z'|Z,A --> R|Z'

See current working paper in paper/main.pdf

To train, first collect and clean data by running:

python gym_data.py 
python preprocess.py

Then train the model by setting TRAIN = True in run.py and run

python run.py

To simply test the model with the newest saved model, set TRAIN = False in run.py and run

python run.py

This will print an example state transition and reward
