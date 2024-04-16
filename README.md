# CSC413 Course Project

- Install `virtualenv`: `pip install virtualenv`
- Create virtualenv: `python -m venv venv`
- Activate venv:
    - Windows: `venv\Scripts\activate`
    - Linux: `source venv/bin/activate`
- Deactivate venv: `deactivate`
- Install dependencies: `pip install -r requirements.txt`

### How to get data:
We are using the kaggle API to obtain the data. Before you run the following commands, make sure you read the [API](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md) documents and generate a kaggle.json file from your kaggle account.
- Install data: `kaggle competitions download -c planttraits2024`

### How to run on remote:
set up the virtual enviornment and then run this command
```
 srun -p csc401 --gres gpu python model.py
```