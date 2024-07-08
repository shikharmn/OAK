# Example run: ./run_pub.sh LF-WikiSeeAlsoTitles-320K YOUR_RUN_NAME 1 0,1,2,3
BASE_DIR="/data"
DATASET=$1
DATA_DIR="${BASE_DIR}/G_Datasets/${DATASET}"
VERSION=$2

LOGGING=$3
if [ $LOGGING  -eq  0 ]
then
    export WANDB_MODE=disabled
fi

echo "Version ${VERSION} on dataset ${DATASET} and WandB logging ${LOGGING} with base directory ${BASE_DIR}."

mkdir -p ${DATA_DIR}/OrganicBERT/${VERSION}

CUDA_VISIBLE_DEVICES=$4 python -W ignore -u main.py \
ngame=False \
regularizer=True \          # Set False for NGAME
base_dir=${BASE_DIR} \
version=${VERSION} \
training_devices=[0,1] \
clustering_devices=[1] \
data=${DATASET} | tee "${DATA_DIR}/OrganicBERT/${VERSION}/debug.log"

rsync --progress -r "tmp/${VERSION}/*" "${DATA_DIR}/OrganicBERT/${VERSION}"