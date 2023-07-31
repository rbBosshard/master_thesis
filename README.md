## _pytcpl_: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository of my master's thesis.

_pytcpl_ is a streamlined Python package that incorporates the **mc4** and **mc5** levels of
[tcpl](https://github.com/USEPA/CompTox-ToxCast-tcpl), 
providing concentration-response curve fitting functionality based on [tcplfit2](https://github.com/USEPA/CompTox-ToxCast-tcplFit2).
It utilizes the [Invitrodb version 3.5 release](https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=355484&Lab=CCTE)
as its backend database.

### (Optional) Use conda environment 
- `conda create --name ml`
- `conda activate ml`
- `conda install pip`

### Install dependencies
  - `pip install -r requirements.txt`

### Run ML for assay endpoint (single aeid)
- Goto [ml.ipynb](ml/ml.ipynb) for running the pipeline (jupyter notebook)
- Goto [config_ml.yaml](config/config_ml.yaml) for customizing ML pipeline behaviour





