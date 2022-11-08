#!/bin/bash -ex
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


OUTPUT_DIR='../dowhy-docs'

#
# Cache existing docs
#
[ -d ${OUTPUT_DIR} ] && rm -rf ${OUTPUT_DIR}
git clone --branch gh-pages https://github.com/py-why/dowhy.git ${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}/.git

#
# Build <0.9 Versions using RTD Theme
#
echo "Building legacy versions (<0.9)"
[ -f source/conf.py ] && rm -rf source/conf.py
cp source/conf-rtd.py source/conf.py
cp source/_templates/versions-rtd.html source/_templates/versions.html
poetry run sphinx-multiversion --dump-metadata source ${OUTPUT_DIR}
# We expect an error with ret-code=2 when SMV cannot find a version to build
set +e
poetry run sphinx-multiversion source ${OUTPUT_DIR}
retVal=$?
set -e

if [[ $retVal -ne 0 ]] && [[ $retVal -ne 2 ]]; then
    echo "error generating documentation"
    exit $retVal
fi

#
# Build >= 0.9 and main branch using Pydata THeme
#
echo "Building versions (>=0.9) and main branch"
rm source/conf.py
cp source/conf-pydata.py source/conf.py
cp source/_templates/versions-pydata.html source/_templates/versions.html
poetry run sphinx-multiversion --dump-metadata source ${OUTPUT_DIR}
poetry run sphinx-multiversion source ${OUTPUT_DIR}

#
# Create the top-level index.html
#
STABLE_VERSION=$(git describe --tags --abbrev=0 --match='v*')

echo "<html>
    <head>
        <meta http-equiv="'"'"refresh"'"'" content="'"'"0; url=./${STABLE_VERSION}"'"'" />
    </head>
</html>" > ${OUTPUT_DIR}/index.html
