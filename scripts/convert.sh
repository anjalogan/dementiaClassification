#!/bin/bash

headDir="$1/*"
for subject in $headDir; do
  subjectId=${subject##*/}
  subjectDir="$subject/*"
  for scan in $subjectDir; do
    scanType=${scan##*/}
    if ! [[ $scanType == *"Calibration"* ]] && ! [[ $scanType == *"calibration"* ]]; then
      scanDir="$scan/*"
      for date in $scanDir; do
        dateDir="$date/*"
        for dicomFolder in $dateDir; do
          cd $dicomFolder
          file=$(ls | head -n 1)
          newFileName="$subjectId::$scanType::${date##*/}.nii"
          mri_convert -it dicom -ot nii $dicomFolder/$file $2/$newFileName > /dev/null
        done
      done
    fi
  done
done
