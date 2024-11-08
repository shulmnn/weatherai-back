from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import httpx
import os

load_dotenv()

WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

class WeatherData(BaseModel): 
  latitude : str
  longitude : str

async def fetchWeather(latitude, longitude):
  async with httpx.AsyncClient() as client:
    response = await client.get(
      WEATHER_API_URL, 
      params={
        "key": WEATHER_API_KEY,
        "q": f"{latitude},{longitude}",
        "aqi": "no",
      }
    )
    data = response.json()
    return data

assistant = client.beta.assistants.create(
  instructions="Hello, you are a weather helper bot. You will receive longitude and latitude from user and you need to use function calling to receive some data. You will need to analyze it and make some assumptions about current weather and how should user dress and what the weather like now. Always answer clearly and choose measurement system depending on country.",
  model='gpt-4o-mini',
  tools=[{
    "type": "function",
    "function": {
      "name": "get_current_weather_data",
      "description" : "Get current weather data with specific latitude and longitude",
      "parameters": {
        "type": "object",
        "properties": {
          "latitude": {
            "type": "string",
            "description" : "Latitude of the specific place",
          },
          "longitude": {
            "type": "string",
            "description" : "Longitude of the specific place",
          }
        },
        "required": ["latitude", "longitude"]
      }
    }
  }],
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/weather')
async def weather(weatherData: WeatherData):
  latitude = weatherData.latitude
  longitude = weatherData.longitude

  thread = client.beta.threads.create()
  message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=f"What is the weather at {latitude},{longitude}"
  )

  run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
  )

  if run.status == 'completed':
    messages = client.beta.threads.messages.list(
      thread_id=thread.id
    )
    print(messages)
  else:
    print(run.status)
  
  tool_outputs = []

  for tool in run.required_action.submit_tool_outputs.tool_calls:
    if tool.function.name == "get_current_weather_data":
      arguments = json.loads(tool.function.arguments)
      latitude = arguments["latitude"]
      longitude = arguments["longitude"]
      data = await fetchWeather(latitude, longitude)
      tool_outputs.append({
        "tool_call_id": tool.id,
        "output": json.dumps(data)
      })

  if tool_outputs:
    try:
      run = client.beta.threads.runs.submit_tool_outputs_and_poll(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs
      )
      print(run)
      print("Tool outputs submitted successfully.")
    except Exception as e:
      print("Failed to submit tool outputs: ", e)
  else:
    print("No tool outputs to sumbit.")

  if run.status == 'completed':
    messages = client.beta.threads.messages.list(
      thread_id=thread.id
    )
    assistant_message = next(
      (msg for msg in messages.data if msg.role == 'assistant'), None
    )
    print(assistant_message)
    if assistant_message and assistant_message.content:
      text_blocks = assistant_message.content
      full_text = ""
      for block in text_blocks:
        if hasattr(block, 'text') and hasattr(block.text, 'value'):
            full_text += block.text.value + "\n"
      
      return {"response": full_text.strip()}
    else:
      return {"error": "Assistant message not found or has no content."}
  else:
    print(run.status)
    return {"error": "Weather data retrieval not completed."}

