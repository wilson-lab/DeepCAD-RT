#!/bin/bash
#SBATCG -c 4                               # Use 4 CPUs
#SBATCH --partition gpu_quad               # Use a quad GPU
#SBATCH --gres=gpu:rtx8000:1,vram:26G      # GPU to use
#SBATCH --time=4-00:00                     # Runtime in D-HH:MM format
#SBATCH --mem-per-cpu=8G                   # Memory total (for all cores)
#SBATCH -o jobs/deepcadrt_train_%j.out     # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e jobs/deepcadrt_train_%j.err           # File to which STDERR will be written, including job ID (%j)

start=`date +%s`

chosen_plane="${2:-0}"
patch_xy="${3:-8}"
patch_t="${4:-100}"

cd /home/ab714/DeepCAD-RT

module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3 
module load python/3.8.12

/n/cluster/bin/job_gpu_monitor.sh &

echo 'training on dataset: ' $1
echo 'using plane: ' $chosen_plane
echo 'activate deepcad_rt'
source activate deepcad_rt
echo "deepcad-rt test initiated"
python flyg_train.py --datasets_path $1 --chosen_plane $chosen_plane --patch_xy $patch_xy --patch_t $patch_t
echo "deepcad-rt train complete"

end=`date +%s`
runtime=$((end-start))
echo "script completed in: " $runtime " seconds"
