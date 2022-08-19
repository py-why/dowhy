Contributing code
==================================================

Local setup
----------------------------------

To contribute code, basic knowledge of git is required, if you are unfamiliar
with the git workflow, checkout `GitHub's tutorial <https://docs.github.com/en/get-started/quickstart/hello-world>`_.
The following steps allow you to contribute code to DoWhy.

1. Fork the `dowhy main repository <https://github.com/py-why/dowhy>`_
2. Clone this repository to your local machine using

.. code:: shell

   git clone https://github.com/<YOUR_GITHUB_USERNAME>/dowhy

3. Install DoWhy and its requirements. Poetry will create a virtual environment automatically,
but if preferred, it can be created in a different way.
By default, Poetry will install DoWhy in interactive mode.
This way, you can immediately test your changes to the codebase.

.. code:: shell

   cd dowhy
   pip install --upgrade pip
   poetry install -E "plotting causalml docs"

.. note::
   Installing pygraphviz can cause problems on some platforms.
   One way, that works for most Linux distributions is to
   first install graphviz and then pygraphviz as shown below.
   Otherwise, please consult the documentation of `pygraphviz <https://pygraphviz.github.io/documentation/stable/install.html>`_.

.. code:: shell

    sudo apt install graphviz libgraphviz-dev graphviz-dev pkg-config
    pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" \
    --install-option="--library-path=/usr/lib/graphviz/"

4. (optional) add dowhy as an upstream remote to keep your
fork up-to-date with DoWhy's main branch.

.. code:: shell

   git remote add upstream http://www.github.com/py-why/dowhy

You are now ready to make changes to the code base locally.

Pull request checklist
----------------------------------

1. Execute the flake8 linter for breaking warnings and fix all reported problems.

.. code:: shell

  poetry run poe lint

2. Add tests for your new code and execute the unittests to make sure
you did not introduce any breaking changes or bugs.

.. code:: shell

  poetry run poe test

Note that you can also execute those tasks together

.. code:: shell

  poetry run poe verify
A full list of available tasks can be obtained using

.. code:: shell

  poetry run poe -h

The full test suite of DoWhy takes quite long. To speed up development cycles,
you can restrict the tests executed as in the following example.

.. code:: shell

  poetry run pytest -v tests/causal_refuters

3. Once your code is finished and it passes all checks successfully,
commit your changes. Make sure to add an informative commit message and to sign off your
commits (DCO):

.. code:: shell

  git commit --signoff -m "informative commit message"