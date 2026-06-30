#!/bin/bash


EBANDSFilename=$1
kptsFilename=$2
bandsFilename=$3

rm $kptsFilename
rm $bandsFilename

infoString=$(awk -v data="# mband: " '$0 ~ data {print $0}' "$EBANDSFilename")

read num1 num2 num3 num4<<<${infoString//[^0-9]/ }

nkpts=$(($num2))
nband=$(($num1))

for ((i=0;i<$nkpts;i++))
#for ((i=0;i<30;i++))
do
	data="# "$i" "
	kString=$(awk -v data="$data" '$0 ~ data {print $0}' "$EBANDSFilename")
	if [[ $i -lt 10 ]]
	then
		echo "${kString:4}" | tr -d "[]" >> $kptsFilename
		bandString=$(awk -v data=$i"  " 'data == substr($0,1,3) {print substr($0,3)}' "$EBANDSFilename")
		echo $bandString >> $bandsFilename
	elif [[ $i -lt 100 ]]
	then
		echo "${kString:5}" | tr -d "[]" >> $kptsFilename
		bandString=$(awk -v data=$i"  " 'data == substr($0,1,4) {print substr($0,4)}' "$EBANDSFilename")
		echo $bandString >> $bandsFilename
	elif [[ $i -lt 1000 ]]
	then
		echo "${kString:6}" | tr -d "[]" >> $kptsFilename
		bandString=$(awk -v data=$i"  " 'data == substr($0,1,5) {print substr($0,5)}' "$EBANDSFilename")
		echo $bandString >> $bandsFilename
	elif [[ $i -lt 10000 ]]
	then
		echo "${kString:7}" | tr -d "[]" >> $kptsFilename
		bandString=$(awk -v data=$i"  " 'data == substr($0,1,6) {print substr($0,6)}' "$EBANDSFilename")
		echo $bandString >> $bandsFilename
	fi
done

