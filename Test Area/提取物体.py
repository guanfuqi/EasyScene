import json
with open("../SceneGraph.json", "r") as f:
    SceneGraph = json.load(f)
Objects:list = [object for object in SceneGraph['objects']]
ObjectNames:list = [object['object'] for object in Objects]
print(ObjectNames)
# for name in ObjectNames:
#     print(name)


