#
# This dockerfile was used to generate darthtrevino/dowhy-docs, used in the 
# build-and-publish-docs.yml workflow
#
FROM ghcr.io/py-why/dowhy-example-notebooks-deps

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install --yes --quiet curl
RUN curl -LsSf https://astral.sh/uv/0.5.2/install.sh | sh

# Install Node for Documntation Tooling
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - 
RUN apt-get install -y nodejs
