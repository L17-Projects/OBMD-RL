#!/bin/bash

del_all() {
	rm -r lammps/eval* 2>/dev/null
	rm model/*.npy 2>/dev/null
	rm test.out 2>/dev/null
	rm test.err 2>/dev/null
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

