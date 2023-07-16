#
# This dockerfile was used to generate darthtrevino/dowhy-docs, used in the 
# build-and-publish-docs.yml workflow
#
FROM ghcr.io/py-why/dowhy-example-notebooks-deps

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install --yes --quiet curl
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.5.1
ENV PATH=$PATH:/root/.local/share/pypoetry/venv/bin

# Install Node for Documntation Tooling
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - 
RUN apt-get install -y nodejs
