#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --job-name=laplace_job
#SBATCH --output=laplace_out_gpu_regs.txt
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1

echo "Node: $SLURM_NODELIST"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 1) Load modules
module load nvhpc

# ======================
# TYPE ↔ REAL_T and DTYPE_STR
# ======================
TYPE="${TYPE:-float}"        # default: float
REAL_T_DEF="float"
NVCC_EXTRA=""

case "$TYPE" in
  float)    REAL_T_DEF="float" ;;
  double)   REAL_T_DEF="double" ;;
  half)     REAL_T_DEF="__half"; NVCC_EXTRA="--expt-extended-lambda -DHAS_HALF" ;;
  float2)   REAL_T_DEF="float2" ;;
  float3)   REAL_T_DEF="float3" ;;
  float4)   REAL_T_DEF="float4" ;;
  double2)  REAL_T_DEF="double2" ;;
  double3)  REAL_T_DEF="double3" ;;
  double4)  REAL_T_DEF="double4" ;;
  half2)    REAL_T_DEF="__half2"; NVCC_EXTRA="--expt-extended-lambda -DHAS_HALF" ;;
  *)
    echo "Invalid TYPE: $TYPE"
    echo "Use: float | double | half | float2 | float3 | float4 | double2 | double3 | double4 | half2"
    exit 1;;
esac

# Map TYPE to a human-readable dtype string for CSV/logs
DTYPE_STR="fp64"
case "$TYPE" in
  float)  DTYPE_STR="fp32" ;;
  double) DTYPE_STR="fp64" ;;
  half)   DTYPE_STR="fp16" ;;
  *)      DTYPE_STR="$TYPE" ;;  # vector/vectorized types kept as-is
esac

# 2) Build (A100)
nvcc -O3 -arch=sm_80 $NVCC_EXTRA -DREAL_T=$REAL_T_DEF -DDTYPE_STR="\"$DTYPE_STR\"" laplace2d_2.cu -o laplace -Xptxas=-v

# 3) One-shot sweep (una sola ejecución barre muchas combinaciones)
#    Edita estas listas si quieres otro espacio de búsqueda:
BX_LIST=${BX_LIST:-"32,64,128,256,512,1024"}
BY_LIST=${BY_LIST:-"1,2,4,8,16,32"}

# Limpia CSV antes (el binario maneja cabecera/filas)
: > tune.csv

# Corre una sola vez; el binario hará el loop interno
srun ./laplace --bx-list="$BX_LIST" --by-list="$BY_LIST"
