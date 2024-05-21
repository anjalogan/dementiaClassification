#!/bin/bash
cd $1
for file in $(ls .); do
  subject_and_type=${file%::*}
  date_with_ext=${file##*::}
  date=${date_with_ext%.*}
  subject=${subject_and_type%::*}
  type=${subject_and_type##*::}
  if ! [[ $type == *"AAHead_Scout"* ]] && ! [[ $type == *"SmartBrain"* ]]; then
    antsRegistrationSyNQuick.sh -d 3 -n 8 -f $2 -m $file -o $subject--$date-- -t r
    mv $subject--$date--Warped.nii.gz $3
    rm $subject--$date--InverseWarped.nii.gz
    rm $subject--$date--0GenericAffine.mat
  fi

done
