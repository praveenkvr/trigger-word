## About

This project includes notebook to train a ML model for keyword detection on a custom word.

THis project is influenced by Courseras Deep learning specialization project for detecting trigger word but includes additional steps to train on custom word and a prediction script.

The word "Hatch" as been used as the trigger word during training and the weights included are for the same.

to test run the following

use the trquirements.txt file to install all dependencies required for prediction.

```python
python predict.py
```

This will start listening through the default michrophone on your machine and play a "chime" sound when it hears the trigger word.