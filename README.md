## _pytcpl_: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository of my master's thesis.
___

### (Optional) Use conda environment 
- `conda create --name ml`
- `conda activate ml`
- `conda install pip`

### Install dependencies
  - `pip install -r requirements.txt`

### Run ML for assay endpoint (single aeid)
- Goto [ml.ipynb](ml/src/ml.ipynb) for running the pipeline (jupyter notebook)
- Goto [config_ml.yaml](config/config_ml.yaml) for customizing ML pipeline behaviour

### Run streamlit
```bash
streamlit run src/app.py --server.address="localhost"
```






