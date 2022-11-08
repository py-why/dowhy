#!/bin/bash -e
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCS_ROOT='../dowhy-docs'
# To change the build target, specify the DOCS_VERSION environment variable. (e.g. DOCS_VERSION=v0.8)
TARGET_BUILD=${DOCS_VERSION:-main}
OUTPUT_DIR="${DOCS_ROOT}/${TARGET_BUILD}"
STABLE_VERSION=$(git describe --tags --abbrev=0 --match='v*')

echo "Building docs for version ${TARGET_BUILD} into ${OUTPUT_DIR}, stable version is ${STABLE_VERSION}"

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
