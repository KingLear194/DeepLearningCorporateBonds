#!/bin/bash
#
#SBATCH --job-name=gan_mp_EJNPort
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jed169@pitt.edu
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48gb
#SBATCH --time=5-20:00:00
#SBATCH --chdir=/ix/jduraj/Bonds/Code
#SBATCH --output=/ix/jduraj/Bonds/Dump/out_%x_%j.out

# change directory to the coding dir
cd /ix/jduraj/Bonds/Code/NN;
# activate environment
source activate_env.sh;

now="$(date "+%d-%m-%H%M")";
SECONDS=0;
project_name='GAN_MP_EJNPort';
jobnum=6;  
modelnums=(1 2);
nrepochs=30; 
batchsizes=(256 512 1024 2048);
l1regs=("none" 0.01);
l2regs=("none" 0.01);
random_seed=0;
config_file_path="/ix/jduraj/Bonds/Data/current_train_val_test/$project_name/config_file_$jobnum.json";

echo "Project name is $project_name";
for modelnr in "${modelnums[@]}"
	do
		for batchsize in "${batchsizes[@]}"
					do
						for l1reg in "${l1regs[@]}"
							do
								for l2reg in "${l2regs[@]}"
									do
    	CUBLAS_WORKSPACE_CONFIG=:16:8 python one_shot_gan.py -modelnr $modelnr -epochs $nrepochs -b $batchsize -lr_sdf 0.0001 -lr_mom 0.0001 -l1reg_sdf $l1reg -l1reg_mom $l1reg -l2reg $l2reg -proj $project_name -rs $random_seed -jobnr $jobnum -config $config_file_path -time_start $now;		
									done
							done
				done
	done;
									
hyperbest=$(python best_hyper.py -proj $project_name -jobnr $jobnum);
echo "Best hyperparams are $hyperbest";

# ensemble predictions
#Train ensemble models: loop over random states with parameters
randomstates=(19 10 3 15 14);
echo "Now calculating the ensemble members with random states ${randomstates[@]}";

for rstate in "${randomstates[@]}"
	do
		CUBLAS_WORKSPACE_CONFIG=:16:8 python one_shot_gan.py -epochs $nrepochs $hyperbest -proj $project_name -rs $rstate -jobnr $jobnum -config $config_file_path -time_start $now;	
	done;

# ensemble predictions
echo "Ensembling..."
valend=235;
python models_ensembling.py -jobnr $jobnum -proj $project_name -ensmethod "mean" $hyperbest -rs ${randomstates[@]} -valend $valend; #val_size = 0.4 as in the preprocessing file
echo "Finished ensembling models for $project_name.";

duration=$SECONDS;
duration_hrs=$(($duration/3600));
echo "Script took $duration_hrs hours, $((($duration-$duration_hrs*3600)/60)) minutes and $((($duration-$duration_hrs*3600)%60)) seconds to run";

conda deactivate;
