FROM ubuntu:focal

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet \
    dirmngr gnupg apt-transport-https ca-certificates software-properties-common \

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet \
    curl r-base pandoc graphviz libgraphviz-dev libgsl-dev libcurl4-openssl-dev libssl-dev libxml2-dev

RUN apt install  --yes --quiet python3-pip python3 git

RUN Rscript -e 'install.packages("BiocManager")'
RUN Rscript -e 'BiocManager::install(c("bnlearn", "pcalg", "kpcalg"))'
RUN Rscript -e 'install.packages("Rcpp", repos="http://cran.us.r-project.org")'
RUN Rscript -e 'install.packages("Matching", repos="http://cran.us.r-project.org")'
RUN Rscript -e 'install.packages("MASS", repos="http://cran.us.r-project.org")'
RUN Rscript -e 'install.packages("momentchi2", repos="http://cran.us.r-project.org")'
RUN Rscript -e 'install.packages("gert", repos="http://cran.us.r-project.org")'
RUN Rscript -e 'install.packages("devtools", repos="http://cran.us.r-project.org")'
RUN Rscript -e 'devtools::install_github("Diviyan-Kalainathan/RCIT")'
RUN Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz")'

RUN pip install pygraphviz
RUN pip install keras tensorflow rpy2 cdt pcalg torch
RUN python3 -m pip install --upgrade pip
RUN pip install sphinx_rtd_theme nbsphinx
RUN pip install 'git+https://github.com/petergtz/sphinx-multiversion.git@override-version-with-refname'
RUN pip install xgboost
RUN pip install pydata-sphinx-theme

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH=$PATH:/root/.local/share/pypoetry/venv/bin
