Installation
^^^^^^^^^^^^

Installing with pip
-------------------

DoWhy requires Python 3.9 or later. To install, you can use pip or conda.

**Latest Release**

Install the latest `release <https://pypi.org/project/dowhy/>`__ using pip.

.. code:: shell
   
   pip install dowhy
   
**Development Version**

If you prefer the latest dev version, clone this repository and install with `Poetry <https://python-poetry.org/>`__:

.. code:: shell

    git clone https://github.com/py-why/dowhy
    cd dowhy
    pip install --upgrade pip
    pip install poetry
    poetry install -E "plotting"

This installs DoWhy in editable mode together with all development dependencies.

Optionally, if you wish to input graphs in the dot format, then install pydot or pygraphviz:

.. code:: shell

    poetry install -E "pydot"
    poetry install -E "pygraphviz"

For better-looking graphs, you can optionally install pygraphviz. To proceed,
first install graphviz system libraries and then pygraphviz (on Ubuntu and Ubuntu WSL).

.. note::
    Installing pygraphviz can cause problems on some platforms.
    One way that works for most Linux distributions is to
    first install graphviz and then pygraphviz as shown below.
    Otherwise, please consult the documentation of `pygraphviz <https://pygraphviz.github.io/documentation/stable/install.html>`_.

.. code:: shell

    sudo apt install graphviz libgraphviz-dev graphviz-dev pkg-config
    pip install pygraphviz

Installing with Conda
---------------------

Install the latest `release <https://anaconda.org/conda-forge/dowhy>`__ using conda.

.. code:: shell

   conda install -c conda-forge dowhy

If you face "Solving environment" problems with conda, then try :code:`conda update --all` and then install dowhy. If that does not work, then use :code:`conda config --set channel_priority false` and try to install again. If the problem persists, please add your issue `here <https://github.com/py-why/dowhy/issues>`_.


Installing on Azure Machine Learning
------------------------------------

In Azure Machine Learning it is not that straight forward to identify in the terminal window the python (Conda) envornoments used by the notebook. Thus, it is easier to run shell commands from within the notebook. The secret is NOT to use the ! magic but the %.

**Getting the latest release**

In an new python code cell type::

    %pip install dowhy

Or::

    %pip install --force-reinstall --no-cache-dir dowhy

**Getting the dev version**

Clone the repository and install in development mode::

    %pip install git+https://github.com/py-why/dowhy.git

Or install from a local clone::

    %pip install -e /path/to/dowhy

To test the installation::

    import dowhy

This should run with no errors.
