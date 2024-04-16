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
use `pip install virtualenv` to install virtual enviornment.
use `virtualenv 413` & `source 413/bin/activate` to build up your virtual env.
run `pip install -r requirements.txt` to install dependencies.
run this command to run your model.
```
 srun -p csc401 --gres gpu python model.py
```