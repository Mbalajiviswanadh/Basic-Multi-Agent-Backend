## create venv and install packages
1. uv init
2. uv venv
3. uv pip install -r requirements.txt and uv add -r requirements.txt

pg-admin to see the data base tables in postgressSQL

for testing:
1. uv run python -m scripts.tavily_tool_test
2. uv run python -m scripts.flight_tool_test.py
3. uv run python -m scripts.build_graph


## Creating tools in files:
Now creating tools files:
1. flight_tool.py
2. tavily_tool.py

## Create backend i.e agent-graph using langgraph:
in backend.py
Inside this we will create agents for:
1. Flight agent
2. Tavily agent
3. Itinerary agent
4. Final response agent
And also creating a shared memory which is STATE 
And build graph after that

## creating FastAPI, API end points
in app.py

End points:
POST	``/api/travel``	Submit a travel query and get AI-generated results
GET	``/health``	Check whether the backend is running

TODO:
build a nice frontend
1. it should show when and what agents its inoked
2. how many llms it took
3. how much time does it take
4. show the resources it used in the response(links, brief content)