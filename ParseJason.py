## Read json file and unpack keys to global/local variables 

class ParseJson:
    def __init__(self, params):
        self.params = params

def obj_creator(d):
    return ParseJson(d)

with open('home/data.json', 'r') as fp:
    data = json.load(fp, object_hook=obj_creator)

for k, v in data.params.iteritems():
    globals()[k] = v

########## could also use  locals()[k]=v for creating local vars
