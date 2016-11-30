#!/bin/bash
cd data
opencv_createsamples \
    -info squares.dat \
    -vec samples.vec \
    -w 20 \
    -h 20
