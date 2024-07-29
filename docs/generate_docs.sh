#!/bin/bash -eu
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCS_ROOT='../dowhy-docs'
# To change the build target, specify the DOCS_VERSION environment variable. (e.g. DOCS_VERSION=v0.8)
CURRENT_VERSION=${DOCS_VERSION:-main}
CI=${CI:-false}
OUTPUT_DIR="${DOCS_ROOT}/${CURRENT_VERSION}"
STABLE_VERSION=$(git describe --tags --abbrev=0 --match='v*')

echo "Reading Tags..."
readarray -t tags_arr < <(git --no-pager tag)
TAGS=$(printf '%s,' "${tags_arr[@]}")

echo "Building docs for version ${CURRENT_VERSION} into ${OUTPUT_DIR}, stable version is ${STABLE_VERSION}, tags=${TAGS}"

# check for required tooling
echo "Verifying Prerequisites"
which npm
which poetry

#
# Cache existing docs
#
if [ ! -f "${DOCS_ROOT}/index.html" ]; then
    echo "Fetching existing docs..."
    git clone --quiet --branch gh-pages https://github.com/py-why/dowhy.git ${DOCS_ROOT}
    rm -rf ${DOCS_ROOT}/.git
fi

#
# Build Docs
#
export CURRENT_VERSION
export TAGS
if  [ $CI == "true" ]; then
    # Using parallelism is slower in GitHub actions
    echo "Executing sphinx-build (Single-Threaded)"
    poetry run sphinx-build source ${OUTPUT_DIR}
else
    echo "Executing sphinx-build (Parallel)"
    poetry run sphinx-build -j auto source ${OUTPUT_DIR}
fi

#
# Patch Version-Selector Info
#
pushd version_patcher
npm install
npm run execute
popd


#
# Create the top-level index.html
#
echo "Creating top-level index.html"
echo "<html>
    <head>
        <meta http-equiv="'"'"refresh"'"'" content="'"'"0; url=./${STABLE_VERSION}"'"'" />
    </head>
</html>" > ${DOCS_ROOT}/index.html

echo "Docsite ready, listing contents..."

find ${DOCS_ROOT}

echo "finished generate_contents.sh"