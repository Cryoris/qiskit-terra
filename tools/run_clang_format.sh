#! /bin/bash

# this is the style file -- note that this script should
# be run from root, such that this file is correctly found
style=".clang-format"

# get all tracked files in HEAD, and filter for files ending in .c or .h
files=$(git ls-tree --name-only -r HEAD | grep ".*\.[c,h]$")

# apply clang format on all files
for file in $files
do
    clang-format --style="file:$style" -i $file
done
