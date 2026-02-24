CONFIG=$1
GPUS=$2
WORKDIR=$3
PORTNUM=$4

PORT=${PORT:-$PORTNUM}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:5} --work-dir $WORKDIR
