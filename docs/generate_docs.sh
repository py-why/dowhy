#!/bin/bash -e
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR='../dowhy-docs/main'

#
# Cache existing docs
#
if [ ! -f "${OUTPUT_DIR}/index.html" ]; then
    git clone --quiet --branch gh-pages https://github.com/py-why/dowhy.git ${OUTPUT_DIR}
    rm -rf ${OUTPUT_DIR}/.git
fi

#
# Build docs
echo "Configuring sphinx"
cp source/_templates/versions-pydata.html source/_templates/versions.html
echo "Executing sphinx-build"
poetry run sphinx-build source ${OUTPUT_DIR}

#
# Create the top-level index.html
#
STABLE_VERSION=$(git describe --tags --abbrev=0 --match='v*')

echo "<html>
    <head>
        <meta http-equiv="'"'"refresh"'"'" content="'"'"0; url=./${STABLE_VERSION}"'"'" />
    </head>
</html>" > ${OUTPUT_DIR}/index.html
