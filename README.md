Step 1 : Created virtual env'.<br>
python -m venv my<br><br>
Step 2: Create setup.py<br><br>
Step 3: Create requirements.txt<br><br>
Step 4: Create .gitignore file<br><br>
Step 5: Create a src folder with components and pipeline as folder in it<br><br>
Step 6: Create <br>
DataIngestion.py : To import the data<br>
Datatransformation.py : To transform the data<br>
ModelTrainer.py : To train the model and store it in the pickle file<br><br>
Step 7: To ease the process write the general purpose functions in the utils folder<br><br>
Step 8: Create a logging.py file to log the indiaction ini logs<br><br>
Step 9: Create a exception.py to handle a exception throwen in any file<br><br>
Step 10: Now create predition.py and train_pipeline.py in pipeline folder<br><br>
train_pipeline.py : To call all the files from the components folder<br>
predition.py : To get the prediction on the new data using the .pkl file in the artifacts folder<br>

