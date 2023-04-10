import os
import json

with open("/home/v-yuancwang/DiffAudioImg/metadata/vgg_10class.json", "r") as f:
    lists = json.load(f)
new_lists = []
for l in lists:
    if l["text"] in ["toilet flushing", "playing electric guitar"]:
        new_lists.append(l)

with open("/home/v-yuancwang/DiffAudioImg/metadata/vgg_2class.json", "w") as f:
    json.dump(new_lists, f)