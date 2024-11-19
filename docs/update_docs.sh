#!/bin/bash -ex

cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

uv run sphinx-apidoc -f -o source ../dowhy
