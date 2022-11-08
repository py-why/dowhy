#!/bin/bash -e

echo "Current Branch: $(git branch --show-current), Head Ref: $GITHUB_HEAD_REF"

echo "------- Git Branches --------"
git --no-pager branch

echo "------- Git Refs ------------"
git for-each-ref --format "%(objectname)\t%(refname)\t%(creatordate:iso)" refs
