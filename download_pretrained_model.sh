gsutil cp -r gs://ai2-raja-public/scienceworld/sciworld_models/rerun/sciworld_11b_rerun_$1 .
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/rerun/sciworld_3b_rerun_$1/special_tokens_map.json sciworld_11b_rerun_$1
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/rerun/sciworld_3b_rerun_$1/spiece.model sciworld_11b_rerun_$1
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/rerun/sciworld_3b_rerun_$1/tokenizer.json sciworld_11b_rerun_$1
gsutil cp gs://ai2-raja-public/scienceworld/sciworld_models/rerun/sciworld_3b_rerun_$1/tokenizer_config.json sciworld_11b_rerun_$1
