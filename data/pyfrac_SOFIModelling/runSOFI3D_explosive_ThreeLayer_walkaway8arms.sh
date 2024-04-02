#!/bin/bash

mkdir -p outputs
mkdir -p outputs/log
mkdir -p outputs/snap
mkdir -p outputs/su

sofipath=~/Documents/Projects/MicroseismicModelling/SOFI3D-master/bin/sofi3D
mpirun -np 8 ${sofipath} explosive_ThreeLayer_walkaway8arms.json > sofi3D.jout

# Do the snap merge thing
snapmergepath=~/Documents/Projects/MicroseismicModelling/SOFI3D-master/bin/snapmerge
${snapmergepath} explosive_ThreeLayer_walkaway8arms.json

# Clean up the snap files (for memory purposes)
rm -rf ./outputs/snap/**.0*
rm -rf ./outputs/snap/**.1*

# Clean up the distributed models
rm -rf ./inputs/models/**.SOFI3D.**
