Contributing code
==================================================

Local setup
----------------------------------

To contribute code, basic knowledge of git is required, if you are unfamiliar
with the git workflow, checkout `GitHub's tutorial <https://docs.github.com/en/get-started/quickstart/hello-world>`_.
The following steps allow you to contribute code to DoWhy.

#. Fork the `dowhy main repository <https://github.com/py-why/dowhy>`_

#. Clone this repository to your local machine using

   .. code:: shell

      git clone https://github.com/<YOUR_GITHUB_USERNAME>/dowhy

#. Install DoWhy and its requirements. Poetry will create a virtual environment automatically,
   but if preferred, it can be created in a different way.
   By default, Poetry will install DoWhy in interactive mode.
   This way, you can immediately test your changes to the codebase.

   .. code:: shell

      cd dowhy
      pip install --upgrade pip
      poetry install -E "plotting"

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

#. (optional) add dowhy as an upstream remote to keep your
   fork up-to-date with DoWhy's main branch.

   .. code:: shell

      git remote add upstream http://www.github.com/py-why/dowhy

   You are now ready to make changes to the code base locally.

Pull request checklist
----------------------------------

#. Execute the flake8 linter for breaking warnings and fix all reported problems.

   .. code:: shell

     poetry run poe lint

#. Make sure the newly added code complies with the format requirements of `black <https://black.readthedocs.io/en/stable/>`_ and
   `isort <https://pycqa.github.io/isort/>`_.

   .. code:: shell

     poetry run poe format_check

   You can use following commands to fix formatting automatically

   .. code:: shell

     poetry run poe format

#. Add tests for your new code and execute the unittests to make sure
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

#. Once your code is finished and it passes all checks successfully,
   commit your changes. Make sure to add an informative commit message and to sign off your
   commits:

   .. code:: shell

     git commit --signoff -m "informative commit message"

   By including this sign-off step, a commit is enriched with a Developer Certificate of Origin (DCO), containing the author's name and email address.
   The DCO is a lightweight alternative to a CLA and affirms that the author is the source of the committed code and has the right to contribute it to the project.
   For the full text, see `DCO <https://developercertificate.org>`_.

   .. note::
      Note the "--signoff" or shorthand "-s" is obligatory and commits without cannot be merged.
      By default, most IDEs won't include this step within their git integration, so an additional setup may be required.

   In case you made a single commit without adding the required DCO, you can do

   .. code:: shell

     git commit --amend --no-edit --signoff
     git push -f origin <BRANCH_NAME>

   In case of more commits, one way is to squash them together (example for 3 commits)

   .. code:: shell

     git reset --soft HEAD~3
     git commit -s -m "new informative commit message of squashed commit"

   or use a rebase with as many "^" as commits to be changed.

   .. code:: shell

      git rebase --signoff HEAD^^^

#. (advanced) Poetry fixes its dependecies and their version with a poetry.lock file. Poetry's dependencies should be updated regularly by maintainers via

   .. code:: shell

     poetry update

   For most PRs, this is unnecessary. If a PR necessitates a lockfile change, we request that you provide a justification as to why a dependency update was necessary.
