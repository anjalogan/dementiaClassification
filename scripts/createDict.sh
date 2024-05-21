#! /bin/bash

declare -A diagnoses
val=""
while IFS="," read -r image_id subject_id diagnosis remaining
do
  diagnoses[$subject_id]=$diagnosis
  val=$subject_id
done < <(head -n 2 $1)
