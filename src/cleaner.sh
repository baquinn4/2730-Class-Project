#!/bin/sh


tag=".csv"
arg=$1

#if ddir dataset doesnt exits, create dir
if [ ! -d "dataset" ]
then
	mkdir "dataset"
fi

if [ "$arg" = "m" ] || [ "$arg" = "d" ]
then
	if [ "$arg" = "m" ]
	then
		for csvfile in $(find -name "*$tag*")
			do
				#moves variables to dir dataset and suppresses output from mv
				mv $csvfile dataset > /dev/null 2>&1
				echo $csvfile " moved to dir dataset"
			done
	else
		for csvfile in $(find -name "*$tag*")
			do
				rm $csvfile
				echo $csvfile " removed from working directory"
			done
	fi		
else
	echo "cleaner takes only one of two args, m (to move .csv files) or d (to delete .csv files)"
fi
