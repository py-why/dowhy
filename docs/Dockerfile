#
# This dockerfile was used to generate darthtrevino/dowhy-docs, used in the 
# build-and-publish-docs.yml workflow
#
FROM ghcr.io/py-why/dowhy-example-notebooks-deps

RUN DEBIAN_FRONTEND=noninteractive apt install --yes --quiet curl
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=$PATH:/root/.local/share/pypoetry/venv/bin
