#!/bin/bash

# r as in Run
# Reminder: You would run it by typing ./r (Return/Enter)
#
# Quick script to run in various configurions
# uncomment the one you want to use

# Maven
# -----
#mvn compile exec:exec
#mvn clean compile exec:exec
# X mvn clean compile exec:exec -- --start

#mvn clean compile && mvn exec:exec -Dexec.args="--start"

#mvn exec:exec       # regular startup
#mvn compile exec:exec       # compile then start
#mvn clean compile exec:exec       # regular startup
#mvn compile exec:exec@start  # with --start

# Jar
# ---
# Tar file build, easier syntax-wise :-)
#echo Running from .jar:
#java -jar target/opencv-pipeline.jar
#java -jar target/opencv-pipeline.jar -h
#java -jar target/opencv-pipeline.jar --webcam -1
java -jar target/opencv-pipeline.jar \
    /Users/mbennett/Dropbox/dev/image-pipelines/webcam_invert.json \
    --auto_run \
    --fullscreen_node_name Monitor \
    --max_time 10

# Syntax
# ------
# OpenCV Pipeline Editor
# 
# Usage: java -jar opencv-pipeline.jar [options] [pipeline.json]
# 
# Options:
#   -h, --help                     Show this help message and exit
#   -a, --auto_start               Automatically start the pipeline after loading
#   --fullscreen_node_name NAME    Show fullscreen preview of node with given name
#   --max_time SECONDS             Exit after specified number of seconds
# 
# Camera options (override all webcam source nodes):
#   --camera_index INDEX           Camera index (-1 for auto-detect)
#   --camera_resolution RES        Resolution: 320x240, 640x480, 1280x720, 1920x1080
#   --camera_fps FPS               Target frame rate (any number, -1 for default 1fps)
#   --camera_mirror BOOL           Mirror horizontally: true/false
# 
# Save behavior options (for use with --max_time):
#   --autosave_prompt              Show save dialog if unsaved changes (default)
#   --autosave_yes                 Automatically save without prompting
#   --autosave_no                  Exit without saving
# 
# Examples:
#   java -jar opencv-pipeline.jar
#   java -jar opencv-pipeline.jar pipeline.json
#   java -jar opencv-pipeline.jar pipeline.json -a
#   java -jar opencv-pipeline.jar pipeline.json -a --camera_index 2 --camera_resolution 1080p
#   java -jar opencv-pipeline.jar pipeline.json -a --fullscreen_node_name "Monitor" --max_time 60
# 
