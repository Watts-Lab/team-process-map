from googleapiclient import discovery
import json

API_KEY = 'AIzaSyBvApFd8k1e0VaYO9iGmXlXUO8525Ln5X4'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request = {
  'comment': { 'text': 'you are the worst person to ever exist' },
  'requestedAttributes': {'TOXICITY': {}, 'INSULT': {}}
}

response = client.comments().analyze(body=analyze_request).execute()
print(json.dumps(response, indent=2))