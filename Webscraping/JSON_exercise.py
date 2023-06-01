import json, requests

serialized = '{"name":"John", "age":30, "car":null}'
            
# deserialzied is the dictionary that Python gets from JSON
deserialized = json.loads(serialized)

# test
assert deserialized['name'] == 'John'

github_user = 'benstager'
endpoint = f"https://api.github.com/users/{github_user}/repos"

repos = json.loads(requests.get(endpoint).text)

# repos is a list of dictionaries, we iterate through each dictionary to
# find the name of each repository in a specific github account, we could do this for other
# keys
for i in repos:
    for j in i.keys():
        if j == 'name':
            print(i[j])