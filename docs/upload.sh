#!/usr/bin/env bash

# Simple script to copy the prob140 docs to gh-pages branch.
# Run this after `make html` and confirm that the documentation looks correct.

git checkout gh-branches
git merge master -m "Merge branch 'master' into gh-pages"

cp -r _build/html/* ..
cd ..

git add *
git commit -m "Updating documentation"
git push origin master

git checkout master
