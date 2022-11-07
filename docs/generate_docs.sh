#!/bin/bash -ex

cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OUTPUT_DIR='../dowhy-docs'

#
# Build <0.9 Versions using RTD Theme
#
cp source/conf-rtd.py source/conf.py
cp source/_templates/versions-rtd.html source/_templates/versions.html

poetry run sphinx-multiversion --dump-metadata source ${OUTPUT_DIR}
poetry run sphinx-multiversion source ${OUTPUT_DIR}

#
# Build >= 0.9 and main branch using Pydata THeme
#
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
