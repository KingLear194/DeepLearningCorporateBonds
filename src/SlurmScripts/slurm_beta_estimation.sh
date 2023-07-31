#!/bin/bash
#
#SBATCH --job-name=BetaNet_SR_EJNPort
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jed169@pitt.edu
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=48gb
#SBATCH --time=2-20:00:00
#SBATCH --chdir=/ix/jduraj/Bonds/Code
#SBATCH --output=/ix/jduraj/Bonds/Dump/out_%x_%j.out

# change directory to the coding dir
cd /ix/jduraj/Bonds/Code/NN;
# activate environment
source activate_env.sh;

# read masteraddr and masterport 
ip="$(hostname -I | cut -f 1 -d' ')";
port="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";

echo "IP address is $ip and freeport is $port";

now="$(date "+%d-%m-%H%M")";
SECONDS=0;
#echo $now;
origin_project_name='SR_EJNPort';
jobnum=23;

project_name="BetaNet_$origin_project_name";

modelnums=(1 2 3 4);
nodes=1;
gpus=4;
nrepochs=50;
batchsizes=(256 512 1024 2048);
learningrates=(0.005 0.005 0.001 0.0001);
l1regs=("none" 0.001 0.01 0.1);
l2regs=("none" 0.001 0.01 0.1);
random_seeds=(0);
config_file_path="/ix/jduraj/Bonds/Data/current_train_val_test/$project_name/config_file_$jobnum.json";

echo "Project name is $project_name";
echo "origin of labels is $origin_project_name";

for modelnr in "${modelnums[@]}"
	do
		for batchsize in "${batchsizes[@]}"
			do
				for l1reg in "${l1regs[@]}"
					do
						for l2reg in "${l2regs[@]}"
							do
 								for rstate in "${random_seeds[@]}"
									do
	   			CUBLAS_WORKSPACE_CONFIG=:16:8 python labels_one_shot.py -device 'cuda' -masteraddr $ip -masterport $port -modelnr $modelnr -rank 0 -n $nodes -g $gpus -epochs $nrepochs -b $batchsize -lr 0.0001 -l1reg $l1reg -l2reg $l2reg -proj $project_name -rs $rstate -jobnr $jobnum -config $config_file_path -time_start $now;
									done
	   						done
					done
			done
	done;

echo "Calculating best hyperparameters";
hyperbest=$(python best_hyper.py -proj $project_name -jobnr $jobnum -val_crit "mean_squared_error");
echo "Best hyperparams are $hyperbest";

randomstates=(19 10 3 15 14);

echo "Now calculating the ensemble members with random states ${randomstates[@]}";
for rstate in "${randomstates[@]}"
	do
		echo "Ensemble with random state $rstate";
		CUBLAS_WORKSPACE_CONFIG=:16:8 python labels_one_shot.py -masteraddr $ip -masterport $port $hyperbest -rank 0 -n $nodes -g $gpus -epochs $nrepochs -proj $project_name -rs $rstate -jobnr $jobnum -config $config_file_path -time_start $now;
	done;

echo "Ensembling..."
# ensemble predictions
python models_ensembling.py -jobnr $jobnum -proj $origin_project_name -ensmethod "mean" $hyperbest -rs ${randomstates[@]} -valend 392 -val_size 0.4 -betanet_jobnr $jobnum;
echo "Finished ensembling models for $project_name.";

duration=$SECONDS;
duration_hrs=$(($duration/3600));
echo "Script took $duration_hrs hours, $((($duration-$duration_hrs*3600)/60)) minutes and $((($duration-$duration_hrs*3600)%60)) seconds to run";

conda deactivate;
