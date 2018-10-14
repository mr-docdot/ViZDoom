PREFIX=demo_test
python run_eval.py --max-num-procs 4 --methods ours --doom-envs deepmind_small --params "$1" --exp-folder-prefix $PREFIX
bash plot_all.sh ../../experiments/${PREFIX}*/log.out
