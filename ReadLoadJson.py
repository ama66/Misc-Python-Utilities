class ParseJson:
    def __init__(self, params):
        self.params = params

def obj_creator(d):
    return ParseJson(d)


with open('settings.json', 'r') as json_file:
        data = json.load(json_file , object_hook=obj_creator)

for k, v in data.params.items():
  globals()[k] = v
  
  
