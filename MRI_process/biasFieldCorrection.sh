#!/bin/sh
#this bash script use fsl to uniform signal intensity

export FSLOUTPUTTYPE='NIFTI'

echo "processing file from $1: $2"

cd $1
cp $2 $4

cd $4

mv $2 T1.nii

${FSLDIR}/bin/fast -B T1.nii

mv T1_restore.nii $3$2

rm -f $4*


