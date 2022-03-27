gsutil cp -r gs://ai2-raja-public/scienceworld/sciworld_models/11b_sciworld_$1_hf .
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/sciworld_3b_$1/special_tokens_map.json 11b_sciworld_$1_hf
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/sciworld_3b_$1/spiece.model 11b_sciworld_$1_hf
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/sciworld_3b_$1/tokenizer.json 11b_sciworld_$1_hf
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/sciworld_3b_$1/tokenizer_config.json 11b_sciworld_$1_hf
