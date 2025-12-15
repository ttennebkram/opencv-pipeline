#!/bin/bash

# Status of uploading rendered Wikipedia pages

aws s3 ls s3://ml-training-wiki-homography/wiki_training/ --recursive --summarize | tail -5
