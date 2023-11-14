#!/bin/bash
#SBATCG -c 1                               # Use CPUs
#SBATCH --partition gpu_quad               # Use a quad GPU
#SBATCH --gres=gpu:rtx8000:1,vram:26G      # Number to use
#SBATCH --time=0-03:00                     # Runtime in D-HH:MM format
#SBATCH --mem-per-cpu=8G                   # Memory total per core
#SBATCH -o jobs/deepcadrt_test_%j.out      # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e jobs/deepcadrt_test_%j.err      # File to which STDERR will be written, including job ID (%j)

start=`date +%s`

denoise_model="${2:-20230811-1_EK021_7f_infinite_bar_methyl_salicylate}"
pth_dir="${3:-/n/data1/hms/neurobio/wilson/DeepCAD_datasets/pth/}"

cd /home/ab714/DeepCAD-RT

module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3 
module load python/3.8.12

/n/cluster/bin/job_gpu_monitor.sh &

echo 'applying model: ' $denoise_model
echo 'on dataset: ' $1
echo 'activate deepcad_rt'
source activate deepcad_rt
echo "deepcad-rt test initiated"
python flyg_test.py --datasets_path $1 --denoise_model $denoise_model --pth_dir $pth_dir
echo "deepcad-rt test complete"

end=`date +%s`
runtime=$((end-start))
echo "script completed in: " $runtime " seconds"
