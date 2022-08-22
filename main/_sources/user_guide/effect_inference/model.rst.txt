Model a causal problem
-----------------------------
DoWhy creates an underlying causal graphical model for each problem. This
serves to make each causal assumption explicit. This graph need not be
complete---you can provide a partial graph, representing prior
knowledge about some of the variables. DoWhy automatically considers the rest
of the variables as potential confounders.

Currently, DoWhy supports two formats for graph input: `gml <https://github.com/GunterMueller/UNI_PASSAU_FMI_Graph_Drawing>`_ (preferred) and
`dot <http://www.graphviz.org/documentation/>`_. We strongly suggest to use gml as the input format, as it works well with networkx. You can provide the graph either as a .gml file or as a string. If you prefer to use dot format, you will need to install additional packages (pydot or pygraphviz, see the installation section above). Both .dot files and string format are supported.

While not recommended, you can also specify common causes and/or instruments directly
instead of providing a graph.

Supported formats for specifying causal assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Graph**: Provide a causal graph in either gml or dot format. Can be a text file
  or a string.
* **Named variable sets**: Instead of the graph, provide variable names that
  correspond to relevant categories, such as common causes, instrumental variables, effect
  modifiers, frontdoor variables, etc.

Examples of how to instantiate a causal model are in the `Getting Started
<https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_simple_example.ipynb>`_
notebook.

