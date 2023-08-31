#!/bin/bash
#SBATCG -c 4                               # Use 4 CPUs
#SBATCH --partition gpu_quad               # Use a quad GPU
#SBATCH --gres=gpu:rtx8000:1,vram:28G              # GPU to use
#SBATCH --time=4-00:00                     # Runtime in D-HH:MM format
#SBATCH --mem=140GB                        # Memory total (for all cores)
#SBATCH -o jobs/deepcadrt_train_%j.out           # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e jobs/deepcadrt_train_%j.err           # File to which STDERR will be written, including job ID (%j)

start=`date +%s`

module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3 
module load python/3.8.12

/n/cluster/bin/job_gpu_monitor.sh &

echo 'activate deepcad_rt'
source activate deepcad_rt
echo "deepcad-rt test initiated"
python flyg_train_example.py
echo "deepcad-rt train complete"

end=`date +%s`
runtime=$((end-start))
echo "script completed in: " $runtime " seconds"