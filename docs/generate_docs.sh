#!/bin/bash -ex

cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OUTPUT_DIR='../dowhy-docs'

mv source/conf.py source/conf.py.orig
cp source/conf-rtd.py source/conf.py
cp source/_templates/versions-rtd.html source/_templates/versions.html

poetry run sphinx-multiversion --dump-metadata source ${OUTPUT_DIR}
poetry run sphinx-multiversion source ${OUTPUT_DIR}

mv source/conf.py.orig source/conf.py
cp source/_templates/versions-pydata.html source/_templates/versions.html

poetry run sphinx-multiversion --dump-metadata source ${OUTPUT_DIR}
poetry run sphinx-multiversion source ${OUTPUT_DIR}

STABLE_VERSION=$(git describe --tags --abbrev=0 --match='v*')

echo "<html>
    <head>
        <meta http-equiv="'"'"refresh"'"'" content="'"'"0; url=./${STABLE_VERSION}"'"'" />
    </head>
</html>" > ${OUTPUT_DIR}/index.html
