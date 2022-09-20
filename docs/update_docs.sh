#!/bin/bash -ex

cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

poetry run sphinx-apidoc -f -o source ../dowhy
