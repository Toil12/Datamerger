import json

json_path="/home/king/PycharmProjects/DataMerger/Data/3rd_Anti-UAV_train_val_thermal/validation.json"


with open(json_path, "r") as js_file:
    data = json.load(js_file)

for key,value in data.items():
    print(key)
    print(len(value))

