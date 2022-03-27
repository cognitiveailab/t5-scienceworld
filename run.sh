echo "Task $1"
echo "Mode $2"

./download_pretrained_model.sh $2
python main.py --task_num=$1 --lm_path=11b_sciworld_$2_hf --beams=16 --mode=$2 --output_path=out-$2test1