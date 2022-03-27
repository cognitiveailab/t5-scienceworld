# sciworld-t5
Install the environment by
```
mkdir logs
conda create --name sciworld-t5 python=3.9
conda activate sciworld-t5
pip install -r requirements.txt
```
You may need to install pytorch manually to use the correct cuda version.
Set --lm_path to the pretrained T5 weights.
To run the model, use
```
python main.py
```
Check the code for all parameters.
