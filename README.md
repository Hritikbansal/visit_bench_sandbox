# Visit Bench Leaderboard

This repo contains code/instructions for updating the visit-bench leaderboard. Link to leaderboard - [https://huggingface.co/spaces/mlfoundations/VisIT-Bench-Leaderboard](https://huggingface.co/spaces/mlfoundations/VisIT-Bench-Leaderboard)

## Commands to add a get ELO Rating for a Single Image model  

1. We need a model predictions csv file with 4 required columns: "instruction", "instruction_category", "image", "model_name", "prediction" where `model_name` is the model's name. Look at the existing files in the `full_model_predictions`.
2. Add this model prediction file to the `full_model_predictions` folder. Make sure the key used does not overlap with any of the existing keys. 
3. Run `make_pairs.py`. This will generate `all_pairs_with_references_original_set.jsonl` (all possible pairs, which we will not be running) and `[model_name]_new_pairs_with_references.jsonl` (queries that we actually will run in step 5).
4. Unzip `gpt-4_cache.jsonl.zip` to get cached `gpt-4` judgments using `unzip gpt-4_cache.jsonl.zip`.
5. Run 
```python
OPENAI_API_KEY=[YOUR OPENAI KEY] python run_all_queries.py --leaderboard_jsonl leaderboard_submission_model_queries/[model_name]_predictions_new_pairs_with_references.jsonl (from step 3)
``` 
to add gpt-4 judgments with the new model into `gpt-4_cache.jsonl`. \
6. Run: `python evaluate_all_queries.py` which will get the judgements `gpt-4_head2head.json`. \
7. Run: 
```python 
python elo_analysis.py --head2head_file gpt-4_head2head.json
```
which will output the leaderboard.


## Committing to the Leaderboard
Please send your `model predictions` file and the zipped version of the updated `gpt-4_cache.jsonl`, from Step 5 above, to the authors at `yonatanbitton1@gmail.com` and `hbansal10n@gmail.com`.
