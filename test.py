from scipy.spatial import distance
import json
bla = '''{"a":true}'''
output = json.loads(bla)
print(output)
print(bla['a'])