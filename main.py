
import argparse
import os
import re
import time
import torch

from math import ceil
from scienceworld import ScienceWorldEnv
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_input_str_behavior_cloning(task_description, prev_obs, prev_action, cur_obs, cur_look, cur_inv):
    outStr = task_description + ' </s> ' + cur_obs + ' ' + cur_inv + ' ' + cur_look + ' </s> <extra_id_0>' + ' </s> ' + prev_action + ' </s> ' + prev_obs + ' </s>'
    outStr = sanitizeStr(outStr)
    return outStr

def build_input_str_decision_transformer(task_description, prev_obs, prev_action, cur_obs, cur_look, cur_inv, cur_score):
    returns_to_go = 1.0 - float(cur_score)
    returns_to_go = round(returns_to_go, 2)
    outStr = task_description + ' </s>' + str(returns_to_go) + '</s> '+  cur_obs + ' ' + cur_inv + ' ' + cur_look + ' </s> <extra_id_0>' + ' </s> ' + prev_action + ' </s> ' + prev_obs + ' </s>'

    outStr = sanitizeStr(outStr)
    return outStr


def sanitizeStr(inStr):
    out = inStr.replace("\n", " ")
    out = out.replace("\t", " ")
    return out

def post_process_generation(raw_pred):
    ans_match = re.match(r".*<extra_id_0>(.*)<extra_id_1>.*", raw_pred)
    if ans_match is not None:
        result = ans_match.group(1)
    else:
        result = raw_pred

    # remove extra <*>'s left in
    result = result.replace("<", " <")
    out = ""
    for token in result.split(" "):
        if (len(token.strip()) > 0):
            if (token[0] != "<"):
                out += token + " "
    result = out

    return result.strip()

#
#   Valid action alignment
#
def findValidAction(predictions, env, lastActions):
    validActions = env.getValidActionObjectCombinations()

    # Go down the ranked list of LM-generated actions.  If one of them is on the valid action list, choose it.
    for pred in predictions:
        if (pred.strip() in validActions):
            if (pred not in lastActions):
                return pred

    # If not, then try to find the cosine of the best action
    tokensPred = predictions[0].split(" ")
    uniqueTokensPred = set(tokensPred)

    topAction = predictions[0]
    topValue = 0
    for validAction in validActions:
        if (validAction not in lastActions):
            tokensAction = validAction.split(" ")
            uniqueTokensAction = set(tokensAction)

            intersection = uniqueTokensPred.intersection(uniqueTokensAction)
            if (len(intersection) > topValue):
                topAction = validAction
                topValue = len(intersection)

    # Sanitize top action
    topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
    return topAction

# Example user input console, to play through a game.
def T5Model(args):

    # Initialize environment
    env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"], threadNum = 0)
    taskNames = env.getTaskNames()
    taskName = taskNames[args["task_num"]]
    env.load(taskName, 0, args['simplification_str'])

    lm_model = T5ForConditionalGeneration.from_pretrained(args["lm_path"]).eval()

    num_layers = len(lm_model.encoder.block)
    mp_size = args["model_parallelism_size"]
    layers_per_device = ceil(num_layers/mp_size)
    device_map = {i: list(range(i*layers_per_device, min((i+1)*layers_per_device, num_layers))) for i in range(mp_size)}

    # device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7],
    #               1: [8, 9, 10, 11, 12, 13, 14, 15],
    #               2: [16, 17, 18, 19, 20, 21, 22, 23]}

    lm_model.parallelize(device_map)
    tokenizer = T5Tokenizer.from_pretrained(args["lm_path"])

    # Pick which set to evaluate on
    variations = []
    if (args["set"] == "test"):
        variations = list(env.getVariationsTest())
    elif (args["set"] == "dev"):
        variations = list(env.getVariationsDev())
    else:
        print("ERROR: Unknown set to evaluate on (" + str(args["set"]) + ")")
        exit(1)


    # Log output prefix
    if (len(args["output_path"]) > 0):
        args["output_path"] = args["output_path"] + "/"

        # Make path if it doesn't exist
        if (not os.path.exists(args['output_path'])):
            os.makedirs(args["output_path"])

    if args["lm_path"].endswith('/'):
        args["lm_path"] = args["lm_path"][:-1]

    filenameOutPrefix = args["output_path"] + "transformer-" + args["mode"] + "-eval-" + str(args["lm_path"].split('/')[-1]) + "-task" + str(args['task_num'])


    scores = []

    for variation in variations:

        env.load(taskName, variation, args["simplification_str"])

        obs, info = env.reset()
        task_description = env.taskdescription()
        prev_obs = ''
        prev_action = ''

        done = False
        score = 0.0
        step = 0

        # The env has an internal step count, some actions like look around are free
        # however, the t5 model only generates the action "look around", which will result in a dead loop below
        # so the max_steps here is only used to avoid the model generating the same action forever
        max_steps = args["env_step_limit"] * 2

        lastNActions = []

        while not done:
            input_str = ""

            if (args["mode"] == "bc"):
                print ("MODE: Behavior Cloning")
                input_str = build_input_str_behavior_cloning(task_description, prev_obs, prev_action, obs, info['look'], info['inv'])
            elif (args["mode"] == "dt"):
                print ("MODE: Decision Transformer")
                input_str = build_input_str_decision_transformer(task_description, prev_obs, prev_action, obs, info['look'], info['inv'], info['score'])
            else:
                print("Unknown mode: " + str(args["mode"]))
                exit(1)

            print("InputStr: " + input_str)
            input_ids = tokenizer(input_str, return_tensors="pt", truncation=True).input_ids

            sample_outputs = lm_model.generate(
                input_ids.to(device),
                max_length=50,
                diversity_penalty=50.0,
                num_return_sequences=args['beams'],
                num_beams=args['beams'],
                num_beam_groups=args['beams'],
            )
            lm_pred = sample_outputs
            lm_pred_text = tokenizer.decode(lm_pred[0])

            # Take the first prediction that is not "look around"
            print("Top N Predictions:")
            useAction = ""
            predStrs = []
            for i, pred in enumerate(lm_pred):
                text = tokenizer.decode(pred)
                text = post_process_generation(text)
                if ((len(useAction) == 0) and (text.strip() != "look around")):
                    useAction = text
                print("\t" + str(i) + "\t" + str(text) )
                predStrs.append(text)

            print(lm_pred_text)

            # Get valid actions at this point
            getBestValidAction = findValidAction(predStrs, env, lastNActions)

            action = getBestValidAction
            obs, reward, done, info = env.step(action)
            score = info['score']
            if score < 0:
                done = True
                score = 0
            print("Obs: " + obs)

            #print("Input string: " + str(input_str))
            print(f"Variation: {variation}, Step: {step}, Score: {score}, Action: {action}")
            print("")
            step += 1
            if (step >= max_steps) or done:
                break

            prev_obs, prev_action = obs, action
            lastNActions.append(action)
            if (len(lastNActions) > 3):
                lastNActions = lastNActions[-4:]

            print("LastNActions: " + str(lastNActions))

            # Early stopping if we're in a loop
            if (len(lastNActions) >= 4):
                if (len(set(lastNActions)) == 1):
                    print("All actions in history are the same -- model is likely in a loop, stopping early.")
                    break


        # Store results
        env.storeRunHistory(variation, notes = {'mode':args["mode"], 'lm':str(args["lm_path"])} )
        env.saveRunHistoriesBufferIfFull(filenameOutPrefix, maxPerFile=args["max_episode_per_file"])

        scores.append(score)

        print("Run completed...")
        print("Scores: " + str(scores))
        time.sleep(2)

    # Episodes are finished -- manually save any last histories still in the buffer
    env.saveRunHistoriesBufferIfFull(filenameOutPrefix, maxPerFile=args["max_episode_per_file"], forceSave=True)

    avg = sum(scores) / len(scores)
    print("Average score: " + str(avg))

    print("Shutting down server...")
    env.shutdown()

    print("Completed.")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar_path", type=str)
    parser.add_argument("--task_num", type=int, default=0)
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--lm_path", default="lm_model")
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--beams", type=int, default=16)
    parser.add_argument("--max_episode_per_file", type=int, default=1000)
    parser.add_argument("--mode", default="bc")
    parser.add_argument("--set", default="test")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--model_parallelism_size", type=int, default=3)

    args = parser.parse_args()
    params = vars(args)
    return params

#
#   Main
#
def main():
    args = parse_args()
    T5Model(args)

if __name__ == "__main__":
    main()