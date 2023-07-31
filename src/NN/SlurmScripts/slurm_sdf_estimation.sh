#!/bin/bash
#
#SBATCH --job-name=MP_bonds
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jed169@pitt.edu
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48gb
#SBATCH --time=5-24:00:00
#SBATCH --chdir=/ix/jduraj/Bonds/Code/NN
#SBATCH --output=/ix/jduraj/Bonds/Dump/out_%x_%j.out

# change directory to the coding dir
cd /ix/jduraj/Bonds/Code/NN;
# activate environment
source activate_env.sh;

# read masteraddr and masterport 
#ip="$(hostname -I | cut -f 1 -d' ')";
#port="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";

echo "IP address is $ip and freeport is $port";

now="$(date "+%d-%m-%H%M")";
SECONDS=0;

project_name='MP_bonds'; # alternatives: SR_bonds, SR_EJNPort, MP_EJNPort 
jobnum=3; # change jobnum accordingly if training multiple jobs in cluster at same time
modelnums=(1 2 3 4);

nrepochs=50;
batchsizes=(256 512 1024 2048);
learningrates=(0.005 0.001 0.0001);
l1regs=("none" 0.001 0.01 0.1);
l2regs=("none" 0.001 0.01 0.1);
random_seeds=(0);
config_file_path="/ix/jduraj/Bonds/Data/current_train_val_test/$project_name/config_file_$jobnum.json";

echo "Project name is $project_name";

for modelnr in "${modelnums[@]}"
	do
		for lrate in "${learningrates[@]}"
			do
				for batchsize in "${batchsizes[@]}"
					do
						for l1reg in "${l1regs[@]}"
							do
								for l2reg in "${l2regs[@]}"
									do
										for rast in "${random_seeds[@]}"
											do
	   	CUBLAS_WORKSPACE_CONFIG=:16:8 python nolabels_one_shot.py -modelnr $modelnr -epochs $nrepochs -b $batchsize -lr $lrate -l1reg $l1reg -l2reg $l2reg -proj $project_name -rs $rast -jobnr $jobnum -config $config_file_path -time_start $now;
										   done
								   done
	   						done
					done
			done
	done;

echo "Calculating best hyperparameters";
hyperbest=$(python best_hyper.py -proj $project_name -jobnr $jobnum);
echo "Best hyperparams are $hyperbest";

#Train ensemble models: loop over random states with parameters
randomstates=(19 10 3 15 14);
# ensemble predictions
echo "Now calculating the ensemble members with random states ${randomstates[@]}";

for rstate in "${randomstates[@]}"
	do
		CUBLAS_WORKSPACE_CONFIG=:16:8 python nolabels_one_shot.py $hyperbest -epochs $nrepochs -proj $project_name -rs $rstate -jobnr $jobnum -config $config_file_path -time_start $now;
	done;

# ensemble predictions
echo "Ensembling..."
python models_ensembling.py -jobnr $jobnum -proj $project_name -ensmethod "mean" $hyperbest -rs ${randomstates[@]} -valend 392; #val_size = 0.4 as in the preprocessing file
echo "Finished ensembling models for $project_name.";


duration=$SECONDS;
duration_hrs=$(($duration/3600));
echo "Script took $duration_hrs hours, $((($duration-$duration_hrs*3600)/60)) minutes and $((($duration-$duration_hrs*3600)%60)) seconds to run";


