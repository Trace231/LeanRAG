from lean_dojo_v2.agent.lean_agent import LeanAgent

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

agent = LeanAgent(database_path="dynamic_database.json")
agent.setup_github_repository(url=url, commit=commit)
agent.train()
print("bootstrap done")
