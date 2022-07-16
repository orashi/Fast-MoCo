export LC_ALL=en_US

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=fastmoco -n16 --gres=gpu:8 --ntasks-per-node=8  \
python -u -m core.solver.fastmoco_solver --config config.yaml --tcp_port $PORT 2>&1|tee train-$$now.log &