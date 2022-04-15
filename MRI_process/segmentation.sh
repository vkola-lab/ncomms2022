#!/bin/sh

export FSLOUTPUTTYPE='NIFTI'

echo "processing file from $1: $2"

cd $1
cp $2 $3/tmp/

cd $3/tmp/

mv $2 T1.nii

${FSLDIR}/bin/fnirt --in=T1 --cout=map --config=/home/sq/newMRIpipeline/seg_atlas/fnirt_config

fslmaths T1 -subsamp2 T1_2mm

invwarp --ref=T1_2mm --warp=map --out=inv_map

applywarp --ref=T1 --in=/home/sq/newMRIpipeline/seg_atlas/MNI_atlas --warp=inv_map --out=$3/seg/$2 --interp=nn

rm -f $3/tmp/*
