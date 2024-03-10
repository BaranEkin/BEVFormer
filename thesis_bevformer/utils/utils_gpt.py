import json
from openai import OpenAI

CLIENT = OpenAI()
SCENE_JSON_PATH = "./data/nuscenes/v1.0-trainval/scene.json"
STATEMENTS_JSON_PATH = "./thesis_bevformer/data/scene_statements.json"

def generate_statement(scene_caption):
    
    gpt_response = CLIENT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be given an input consisting of comma separated information regarding a traffic scene."
                           "Your task is to reply with simple sentences that describe the situation in the ego vehicle's perspective."
                           "Always refer to yourself as 'the ego vehicle.'"
                           "Avoid adding extra information when you can. Use simple present.",
            },
            {
                "role": "user",
                "content": "Narrow street, lane change, several peds, turn left, avoid parked cars",
            },
            {
                "role": "assistant",
                "content": "The ego vehicle is on a narrow street. The ego vehicle changes lanes. There are several pedestrians around."
                           "The ego vehicle turns left while avoiding parked cars.",
            },
            {
                "role": "user",
                "content": scene_caption,
            },
            
        ],
    )
    return gpt_response.choices[0].message.content

if __name__ == "__main__":
    
    with open(STATEMENTS_JSON_PATH, "w") as statements_json:
        with open(SCENE_JSON_PATH, "r") as scene_json:
            scenes = json.load(scene_json)

        scene_statements = []
        for scene in scenes:
            try:
                scene_statement = {}
                scene_token = scene["token"]
                scene_description = scene["description"]
                statement = generate_statement(scene_description)

                print("SCENE:\t", scene["name"], "____________")
                print("Description:\t", scene_description)
                print("Statement:\t", statement, end="\n\n")

                scene_statement["scene_token"] = scene_token
                scene_statement["scene_description"] = scene_description
                scene_statement["statement"] = statement
                scene_statements.append(scene_statement)
            except:
                break

        json.dump(scene_statements, statements_json, indent=4)

