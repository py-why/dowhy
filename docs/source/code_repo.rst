Code repository
=================

DoWhy is hosted on visualstudio.com. At present, the platform does not allow public access
to the repo.

You can browse the code in a html-friendly format `here <dowhy.html>`_.

To enable access to the code repository, raise a request at http://idweb/ to be added to the
"dowhy" security group. You will be automatically added to the group and then
should be able to access https://amshar.visualstudio.com/DoWhy.

If there are any problems, please email us at `dowhy@microsoft.com
<mailto:dowhy@microsoft.com>`_.

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


