# Installing DoWhy in Azure Machine Learning

## 14 June 2022
### Eli Y. Kling {https://www.linkedin.com/in/elikling/}

In Azure Machine Learning it is not that straight forward to identify in the  terminal window the python (Conda) envornoments used by the notebook. Thus, it is easier to run shell commands from within the notebook. The secrete is NOT to use the ! magic but the %.

### 1. The straight-forward way
In an new python code cell type

`%pip install dowhy`

Or

`%pip install --force-reinstall --no-cache-dir dowhy`

### 2. Getting the dev version
a. Open a new terminal window - it will open pointing to your user folder

b. Create a new folder (if you wish - this is not really necessary)

`mkdir pywhy`

c. To be really pedantic ensure it is fully 'activated'

`chmod 777 pywhy`

d. Get the full path by

```
    cd pywhy
    pwd
```

e. Copy that path you will need it later

f. Clone the repository

`git clone https://github.com/py-why/dowhy`

g. Now open an python notebook and create a new python code cell. Type:

`%pip install -e <path from step d.>`

h. To test the installation:
	    
```
    import numpy as np, pandas as pd
	from dowhy import CausalModel
	import dowhy.datasets
	import networkx as nx
	from dowhy import gcm
```
		
Should run with no errors
	 
