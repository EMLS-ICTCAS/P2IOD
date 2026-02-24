CONFIG=$1
CHECKPOINT=$2
WOORKDIR=$3
GPUS=$4
PORTNUM=$5
PORT=${PORT:-$PORTNUM}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py --config $CONFIG --checkpoint $CHECKPOINT --work-dir $WOORKDIR --launcher pytorch ${@:6}
