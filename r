#!/bin/bash

# Quick script to run in various configurions
# uncomment the one you want to use

#mvn compile exec:exec
#mvn clean compile exec:exec
# X mvn clean compile exec:exec -- --start

#mvn clean compile && mvn exec:exec -Dexec.args="--start"

#mvn compile exec:exec       # regular startup
mvn clean compile exec:exec       # regular startup
#mvn compile exec:exec@start  # with --start
