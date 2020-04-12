# BioBert NER

This package helps to do NER using BioBERT (or any BERT model). There wasn't many packages available to use the BERT 
models in your module as an api. This is a quick fix work. What I meant by is that I have used the same set of files that 
are provided by the BioBert [repo](https://github.com/dmis-lab/biobert) and just added a facade to it. I could have avoided the use of storing to temp files and 
reading it again for processing by just keeping them in memory but my need was to quickly test the BioBERT model so am 
using exactly the same way the BioBERT git [repo](https://github.com/dmis-lab/biobert) uses. 

This is tested on a model trained using PubMed 200K + PMC 270K data.

### Requirements

The code is designed for `Python 3.x`. Though we have tried to provide backward compatibility with `Python 2.x` wherever 
possible, I strongly recommend to use `Python 3.x`.

For all other dependencies, please check the `requirements.txt` file.

### How to use
We also strongly recommend to use a **virtual environment** to run this code to keep it isolated and to avoid any conflicts 
as we used many different libraries that would possibly cause some version conflicts with the underlying libraries that 
they used.

#### Installation

##### Setting up Virtual Environment
[Conda](https://conda.io/) can be used set up a virtual environment with the version of Python required. 
If you already have a Python 3.6 or 3.7 environment you want to use, you can skip to the 'installing via pip' section.
1. [Follow the installation instruction for Conda] (https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda#regular-installation)

2. Create a virtual environment called "biobert_py36" with Python 3.6:
```python
conda create -n biobert_py36 python=3.6
```
or
```python
conda create --name biobert_py36 python=3.6
```

3. Activate the Conda environment
```python
conda activate biobert_py36

Now you can either install the package or can install the required dependecies independetly without installing the package.
```
##### Installing as a package
Installing BioBertNER requires two steps: Downloading the package wheel file from the **bin** folder and installing. 
To install the library, run:
```sh
python -m pip install name_of_the_whl_file.whl
```

#### Using without installation 
Run the below command within the root folder where you have requirements.txt:

```python
pip install -r requirements.txt
```

Please see [here] (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) for more help on Conda virtual environment creation. 

If you run into issues while installing related to pytorch, please visit this site and choose your right system configuration/requirement
and run the command given by them to install `torch` and `torchvision`. For e.g. to install a Non CUDA version on a Windows 
machine, run the below command:
```python
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
``` 
In which case, please go and comment/remove the lines that install torch and torchvision in your `requirements.txt` file.

### Inference
 
 
 ```python
from biobert_ner import BioBertNER
 
text = """
This is a 14-month-old baby boy Caucasian who came in with presumptive diagnosis of Kawasaki with fever for more than 5 days and conjunctivitis, mild arthritis with edema, rash, resolving and with elevated neutrophils and thrombocytosis, elevated CRP and ESR.
"""

# Get the named entities
ents, nes = ner.predict(text)
entities = [ne["text"] for ne in nes]
```

### Building the package
Go tho the source folder where you have the `setup.py`. Run below command to build the package.

```sh
python setup.py sdist bdist_wheel
```

Once the package is built, you will see a `dist` folder and within the folder a `.tar.gz` file and `.whl` file. Run the
below command to install the package

```sh
python -m pip install name_of_the_whl_file.whl
```

You can also download the pre-built pacakge from `bin` folder and run the above command if you do not want to build.


## Contributing and Making Changes

This is a common component and you are free to make any changes to it make it better. Once the changes are tested, you
can build the package using the steps above.