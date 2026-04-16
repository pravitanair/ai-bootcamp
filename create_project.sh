#!/bin/bash

PROJECT_NAME=$1

if [ -z "$PROJECT_NAME" ]; then
  echo "Usage: ./create_project.sh <project-name>"
  exit 1
fi

mkdir $PROJECT_NAME
cd $PROJECT_NAME

mkdir data
touch main.py README.md requirements.txt .env

python3 -m venv venv

echo "Project $PROJECT_NAME created successfully"