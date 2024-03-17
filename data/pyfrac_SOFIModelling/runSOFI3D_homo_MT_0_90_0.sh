#!/bin/bash

mkdir -p outputs
mkdir -p outputs/log
mkdir -p outputs/snap
mkdir -p outputs/su

sofipath=~/Documents/Projects/MicroseismicModelling/SOFI3D-master/bin/sofi3D
mpirun -np 8 ${sofipath} homo_MT_0_90_0.json > sofi3D.jout

# Do the snap merge thing
snapmergepath=~/Documents/Projects/MicroseismicModelling/SOFI3D-master/bin/snapmerge
${snapmergepath} homo_MT_0_90_0.json

# Clean up the snap files (for memory purposes)
rm -rf ./outputs/snap/**.0*
rm -rf ./outputs/snap/**.1*
