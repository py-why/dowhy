Installation
^^^^^^^^^^^^

Installing with pip
-------------------

DoWhy support Python 3.6+. To install, you can use pip or conda. 

**Latest Release**

Install the latest `release <https://pypi.org/project/dowhy/>`__ using pip.

.. code:: shell
   
   pip install dowhy
   
**Development Version**

If you prefer the latest dev version, clone this repository and run the following command from the top-most folder of
the repository.

.. code:: shell
    
    pip install -e .

**Requirements**

If you face any problems, try installing dependencies manually.

.. code:: shell
    
    pip install -r requirements.txt

Optionally, if you wish to input graphs in the dot format, then install pydot (or pygraphviz).


For better-looking graphs, you can optionally install pygraphviz. To proceed,
first install graphviz and then pygraphviz (on Ubuntu and Ubuntu WSL).

.. note::
    Installing pygraphviz can cause problems on some platforms.
    One way that works for most Linux distributions is to
    first install graphviz and then pygraphviz as shown below.
    Otherwise, please consult the documentation of `pygraphviz <https://pygraphviz.github.io/documentation/stable/install.html>`_.

.. code:: shell

    sudo apt install graphviz libgraphviz-dev graphviz-dev pkg-config
    pip install --global-option=build_ext \
    --global-option="-I/usr/local/include/graphviz/" \
    --global-option="-L/usr/local/lib/graphviz" pygraphviz

Installing with Conda
---------------------

Install the latest `release <https://anaconda.org/conda-forge/dowhy>`__ using conda.

.. code:: shell

   conda install -c conda-forge dowhy

If you face "Solving environment" problems with conda, then try :code:`conda update --all` and then install dowhy. If that does not work, then use :code:`conda config --set channel_priority false` and try to install again. If the problem persists, please add your issue `here <https://github.com/microsoft/dowhy/issues/197>`_.


Installing on Azure Machine Learning
------------------------------------

In Azure Machine Learning it is not that straight forward to identify in the terminal window the python (Conda) envornoments used by the notebook. Thus, it is easier to run shell commands from within the notebook. The secret is NOT to use the ! magic but the %.

**Getting the latest release**

In an new python code cell type::

    %pip install dowhy

Or::

    %pip install --force-reinstall --no-cache-dir dowhy

**Getting the dev version**

a. Open a new terminal window - it will open pointing to your user folder

b. Create a new folder (if you wish - this is not really necessary)::

    mkdir pywhy   

c. To be really pedantic, ensure it is fully 'activated'::

    chmod 777 pywhy

d. Get the full path by::

    cd pywhy
    pwd

e. Copy that path you will need it later.

f. Clone the repository::

    git clone https://github.com/py-why/dowhy

g. Now open a python notebook and create a new python code cell. Type::

    %pip install -e <path from step d.>

h. To test the installation::

    import dowhy
    

This should run with no errors.
