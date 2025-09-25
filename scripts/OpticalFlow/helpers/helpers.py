# Purpose: Generic helpers for optical flow computation

import json

def getCellChannelFromJSON(jsonFile):
    with open(jsonFile) as f:
        channelSpecs = json.load(f)["channels"]
    cells_found = False
    for i, channelInfo in enumerate(channelSpecs):
        if channelInfo["name"].startswith("cells"):
            cells = i
            if cells_found:
                print(f"Warning: Multiple channels starting with 'cells' found. Multiple cell channels is not supported. Using channel {i}.")
            print(f"Found cell channel: {i}")
            cells_found = True
    if not cells_found:
        print("Error: No channel starting with 'cells' found in parameters.json.")
        return None
    return cells

def getRockChannelFromJSON(jsonFile):
    with open(jsonFile) as f:
        channelSpecs = json.load(f)["channels"]
    rocks_found = False
    for i, channelInfo in enumerate(channelSpecs):
        if channelInfo["name"].startswith("rocks"):
            rocks = i
            if rocks_found:
                print(f"Warning: Multiple channels starting with 'rocks' found. Multiple rock channels is not supported. Using channel {i}.")
            print(f"Found rock channel: {i}")
            rocks_found = True
    if not rocks_found:
        print("Error: No channel starting with 'rocks' found in parameters.json.")
        return None
    return rocks