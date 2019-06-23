Code repository
=================

DoWhy is hosted on GitHub.

You can browse the code in a html-friendly format `here
<https://github.com/Microsoft/dowhy>`_.

Installation
------------

**Requirements**

DoWhy support Python 3+. It requires the following packages:

* numpy 
* scipy
* scikit-learn
* pandas
* pygraphviz (for plotting causal graphs)
* networkx  (for analyzing causal graphs)
* matplotlib (for general plotting)
* sympy (for rendering symbolic expressions)


On Ubuntu WSL/Windows 10, the following lines will install dependencies::
    
    pip3 install numpy
    pip3 install sklearn
    pip3 instlal pandas
    sudo apt install graphviz libgraphviz-dev graphviz-dev pkg-config
    ## from https://github.com/pygraphviz/pygraphviz/issues/71
    pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" \
     --install-option="--library-path=/usr/lib/graphviz/"
    pip3 install networkx
    pip3 install matplotlib
    pip3 install sympy


Questions/Feedback
------------------
For any general questions or feedback, you can reach us at the causal inference
distribution list `causalinference@microsoft.com <mailto:causalinference@microsoft.com>`_.


