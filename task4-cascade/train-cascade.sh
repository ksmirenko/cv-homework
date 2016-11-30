#!/bin/bash
cd data
opencv_traincascade \
    -data cascade/ \
    -vec samples.vec \
    -bg triangles.dat \
    -numStages 10 \
    -minhitrate 0.999 \
    -maxFalseAlarmRate 0.5 \
    -numPos 600 \
    -numNeg 800 \
    -w 20 \
    -h 20 \
    -featureType LBP \
    -mode ALL \
    -precalcValBufSize 4096 \
    -precalcIdxBufSize 4096 \
    -numThreads 24
