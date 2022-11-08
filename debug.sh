#!/bin/bash -ex

echo "Current Branch: $(git branch --show-current), Head Ref: $GITHUB_HEAD_REF"

git for-each-ref --format "%(objectname)\t%(refname)\t%(creatordate:iso)" refs
