#!/bin/bash

del_all() {
	rm -r lammps/env* 2>/dev/null
	rm -r lammps/eval* 2>/dev/null
	rm log.lammps 2>/dev/null
	rm -r models 2>/dev/null
	rm -r model 2>/dev/null 
	rm -r logs_SAC_protein 2>/dev/null
	rm -r checkpoints_SAC 2>/dev/null
	rm training_duration.txt 2>/dev/null
	rm timestep.txt 2>/dev/null
	rm rl.out 2>/dev/null
	rm rl.err 2>/dev/null
}

cases () {
	case $1 in 
		[yY]) 
		printf "\n"
		del_all
		printf "\nCleaned!\n" ;;
		[nN])
		printf "\nExiting without cleaning\n" ;;
		*)
		printf "\nInvalid option '$1'. Use y, Y, n or N.\n" ;;
	esac
}

read -n 1 -p $'Are you sure you want to clean all? [yY/nN] \n' reply; 

cases $reply
