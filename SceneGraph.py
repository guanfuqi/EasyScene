from openai import OpenAI
import json
from datetime import datetime
import os

class SceneGraph(object):
    '''
    主要方法：
        gen_scene_graph(text):
            针对源文本生成场景图，返回.json文本
        gen_new_text(self,scene_graph_json:str):
            生成新文本，返回新文本
        save_scene_graph(self,path):
            保存场景图在指定路径
        extract_objects_names():
            提取物体列表，返回一个字符串序列
        extract_object_prompt(self, name:str):
            提取指定物体的提示词，返回字符串(例如”窗帘，透光的，蓝色的”）
        例子：
        生成新场景图
            scene_graph = SceneGtaph("a bedroom with a double bed",ds_key = ..., id = ...)
            ds_key：你的deepseek key
            id：输出文件夹的名字
        加载已存在的场景图：
             scene_graph = SceneGtaph(exist: bool = True,  exist_path:str = id）
             exist_path:输出文件夹路径,该目录下应有一个场景图json文件

    '''
    text2graph_system_prompt = """You are a scene graph generator. Convert the given text into JSON format, then add some detailed attributes for eash object in the scene gtaph.
Strictly follow these rules:
1. Use ONLY valid JSON syntax
2. Exclude Markdown formatting
3. Preserve original language terms
4. Include ALL mentioned entities
5. Maintain this exact structural hierarchy
6. For indoor scenes, extract the objects in them and ignore the higher-level terms such as 'bedroom' and 'kitchen'

example output:
{
    "objects": {
        "desk lamp" : {
            "attributes": ["orange", "metal", "tilted"],
            "relations": [{"target":"desk", "relation":"illuminating"}]
        },
            "desk" : {
            "attributes": ["walnut finish", "1.2m length"]
        }
    }
}
"""

    graph2text_system_prompt = """Act as a scene reconstruction engine. Generate a short belief sentence from scene graph JSON data. The sentence includes all the objects and their corresponding attributes.

Example Input:
{
    "objects": {
        "desk lamp" : {
            "attributes": ["orange", "metal", "tilted"],
            "relations": [{"target":"desk", "relation":"illuminating"}]
        },
            "desk" : {
            "attributes": ["walnut finish", "1.2m length"]
        }
    }
}

Example Output:
"A tilted orange metal desk lamp casts warm light across the walnut-finished writing desk, its 1.2-meter surface revealing subtle wood grain patterns under the golden illumination."
"""
    default_key = "sk-04cbd40be8f84c11a708796474316d93"

    def __init__(self, old_text: str, ds_key: str = default_key, id: str = datetime.now().strftime('%Y%m%d%H%M%S'), *, exist: bool = False,  exist_path:str = None):
        if exist:
            for filename in os.listdir(exist_path):
                # 筛选以 "text" 开头的文件，并排除子目录
                if filename.startswith("new_text") and os.path.isfile(os.path.join(exist_path, filename)):
                    # 拼接完整文件路径
                    filepath = os.path.join(exist_path, filename)
                    # 打开文件并操作（例如读取内容）
                    with open(filepath, "r") as file:
                        self.new_text = file.read()
            
            for filename in os.listdir(exist_path):
                # 筛选以 "text" 开头的文件，并排除子目录
                if filename.startswith("scene_graph") and os.path.isfile(os.path.join(exist_path, filename)):
                    # 拼接完整文件路径
                    filepath = os.path.join(exist_path, filename)
                    # 打开文件并操作（例如读取内容）
                    with open(filepath, "r") as file:
                        self.scene_graph_json = file.read()
            self.scene_graph:dict = json.loads(self.scene_graph_json)
        else:
            self.ds_key = ds_key
            self.client = OpenAI(api_key = self.ds_key, base_url = "https://api.deepseek.com")
            self.old_text:str = old_text
            self.scene_graph_json = self.gen_scene_graph(self.old_text)
            self.scene_graph:dict = json.loads(self.scene_graph_json)
            self.new_text = self.gen_new_text(self.scene_graph_json)
            self.save_path = f'./output/{id}'

            os.makedirs(self.save_path, exist_ok = True)
            self.save_scene_graph(os.path.join(self.save_path, 'scene_graph.json'))
            with open(os.path.join(self.save_path, 'new_text.txt'), 'w') as f:
                f.write(self.new_text)
            
    def gen_scene_graph(self,old_text):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SceneGraph.text2graph_system_prompt},
                {"role": "user", "content": old_text},
            ],
            max_tokens=1024,
            temperature=1.0,
            stream=False
        )
        return response.choices[0].message.content

    def gen_new_text(self,scene_graph_json:str):
        response = self.client.chat.completions.create(
            model = "deepseek-chat",
            messages = [
                {"role" : "system", "content" : SceneGraph.graph2text_system_prompt},
                {"role": "user", "content": scene_graph_json}
            ],
            max_tokens = 1024,
            temperature = 1.0,
            stream = False
        )
        return response.choices[0].message.content

    def save_scene_graph(self,path):
        if self.scene_graph_json != None:
            with open(path, 'w') as f:
                f.write(self.scene_graph_json)
        else:
            raise Exception("scene_graph_json:str is None!")

    def extract_objects_names(self):
        if self.scene_graph != None:
            names = []
            objects:dict = self.scene_graph['objects']
            for name in objects.keys():
                names.append(name)
            return names
        else:
            raise Exception("scene_graph:dect is None!")

    def extract_object_prompt(self, name:str):
        objects = self.scene_graph['objects']
        attributes = objects[name]['attributes']
        prompt = ','.join([name] + attributes)
        return prompt

if __name__ == '__main__':
    with open('input/text.txt', 'r') as f:
        text = f.read()
    scene_graph = SceneGraph(
        old_text = text,
        ds_key = 'sk-04cbd40be8f84c11a708796474316d93')
    names = scene_graph.extract_objects_names()
    print(names)
    print(scene_graph.new_text)