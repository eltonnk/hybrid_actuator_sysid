## Setup
Open a terminal and enter
```
py -m venv env
```

Then launch the virtual environment and type 

```
pip install -e pysysid/ --config-settings editable_mode=strict
```

## Update 

To update the pysysid to its latest changes, run 

```
git submodule update --remote
```


## How to 

Put you csv file in a `data` folder in this repo.

Run the `hybrid_id.py` file in this repo.

Hybrid actuators parameters corresponding to the given dataset will be identified.