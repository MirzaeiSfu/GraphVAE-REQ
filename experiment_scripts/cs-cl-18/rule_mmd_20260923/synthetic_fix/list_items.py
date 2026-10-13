import json, os, sys
for it in json.load(open(sys.argv[1] + "/inputs.json"))["items"]:
    if os.path.exists(it["path"]):
        print("%s_seed_%s\t%s" % (it["method"], it["seed"], it["path"]))
