#!/bin/bash -e
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCS_ROOT='../dowhy-docs'
OUTPUT_DIR="${DOCS_ROOT}/main"
STABLE_VERSION=$(git describe --tags --abbrev=0 --match='v*')

#
# Cache existing docs
#
if [ ! -f "${DOCS_ROOT}/index.html" ]; then
    git clone --quiet --branch gh-pages https://github.com/py-why/dowhy.git ${DOCS_ROOT}
    rm -rf ${DOCS_ROOT}/.git
fi

#
# Build docs
echo "Executing sphinx-build"
poetry run sphinx-build source ${OUTPUT_DIR}

#
# Create the top-level index.html
#

echo "<html>
    <head>
        <meta http-equiv="'"'"refresh"'"'" content="'"'"0; url=./${STABLE_VERSION}"'"'" />
    </head>
</html>" > ${DOCS_ROOT}/index.html
