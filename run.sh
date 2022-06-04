echo "Task $1"
echo "Mode $2"

./download_pretrained_model.sh $2
python main.py --task_num=$1 --lm_path=sciworld_11b_rerun_$2 --beams=16 --mode=$2 --output_path=out-$2test1