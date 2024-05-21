#!/bin/bash

# arg1: directory with nii files
# arg2: location of patient diagnoses csv
# arg3: location of slice.py
# arg4: location to store the final pngs

bar_size=40

show_progress_bar () {
  item=$1
  total=$2
  # calculate the progress in percentage
  percent=$(bc <<< "scale=2; 100 * $item / $total" )
# The number of done and todo characters
  done=$(bc <<< "scale=0; $bar_size * $percent / 100" )
  todo=$(bc <<< "scale=0; $bar_size - $done" )
  done_sub_bar=$(printf "%${done}s" | tr " " "#")
  todo_sub_bar=$(printf "%${todo}s" | tr " " "-")

  echo -ne "\rProgress : [${done_sub_bar}${todo_sub_bar}] ${percent}%"
}

echo "Processing files in $1"

# Create directories to store the final pngs
cd $4
mkdir -p CN MCI AD
echo "Storage directories created"

# Create dictionary to determine diagnosis of the patient
declare -A diagnoses
while IFS="," read -r image_id subject_id diagnosis remaining
do
  # Trim quotes from the text
  subject_id=${subject_id%\"*}
  subject_id=${subject_id#\"*}
  diagnosis=${diagnosis%\"*}
  diagnosis=${diagnosis#\"*}
  # Put in dictionary
  diagnoses[$subject_id]=$diagnosis
done < $2

echo "Dictionary built"
echo "Beginning file processing"

# Strip the skulls and take slices
cd $1
val=0
total=$(ls -p . | grep -v / | wc -l)
for file in $(ls -p . | grep -v /); do
  val=$((val+1))
  show_progress_bar $val $total
  id=${file%%--*}
  bet $file stripped$file -f 0.4
  python3 $3 stripped$file $4 ${diagnoses[$id]}
  rm stripped$file
done
echo ""
