Step 1 : Created virtual env 
python -m venv my
Step 2: Create setup.py
Step 3: Create requirements.txt
Step 4: Create .gitignore file
Step 5: Create a src folder with components and pipeline as folder in it
Step 6: Create 
DataIngestion.py : To import the data
Datatransformation.py : To transform the data
ModelTrainer.py : To train the model and store it in the pickle file
Step 7: To ease the process write the general purpose functions in the utils folder
Step 8: Create a logging.py file to log the indiaction ini logs
Step 9: Create a exception.py to handle a exception throwen in any file
Step 10: Now create predition.py and train_pipeline.py in pipeline folder
train_pipeline.py : To call all the files from the components folder
predition.py : To get the prediction on the new data using the .pkl file in the artifacts folder

