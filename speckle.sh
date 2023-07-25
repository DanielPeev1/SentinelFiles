#!/bin/bash


files=$(ls ./preprocessed-s1/)

for file in ${files}
do
  date=${file}
  satFiles=$(ls ./preprocessed-s1/"${file}"/)
  
  for satFile in ${satFiles}
  do
    sat=$(echo ${satFile} | cut -d '-' -f 1)
    pol=$(echo ${satFile} | cut -d '-' -f 2 | cut -d '.' -f 1 | tr '[:lower:]' '[:upper:]')

    if [ ! -d "./LeeSigma-s1/${date}" ]; then
      mkdir "./LeeSigma-s1/${date}"
    fi


    /Applications/snap/bin/gpt ./LeeSigma.xml -Pinput="./preprocessed-s1/${file}/${satFile}" -Pspeck_pol="Sigma0_${pol}_db" -PdbPol="Sigma0_${pol}" -Poutput="./LeeSigma-s1/${date}/${sat}-${pol}"
  done
done