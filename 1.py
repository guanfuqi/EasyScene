from SceneGraph import SceneGraph
with open('input/text.txt', 'r') as f:
    text = f.read()

scene_graph = SceneGraph(text, exist = True, exist_path = 'output/20250312193327')

class_names = scene_graph.extract_objects_names()
print(class_names)