#!/bin/bash

mkdir -p ./miss_ddl
mkdir -p ./train_time
mkdir -p ./used_gpus
mkdir -p ./used_gpus_infer
mkdir -p ./used_gpus_train
mkdir -p ./dist
mkdir -p ./avg_lat

rm -f ./miss_ddl/*
rm -f ./train_time/*
rm -f ./used_gpus/*
rm -f ./used_gpus_infer/*
rm -f ./used_gpus_train/*
rm -f ./dist/*
rm -f ./avg_lat/*