from openai import OpenAI

CLIENT = OpenAI()

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
                "content": "The ego vehicle is on a narrow street. The ego vehicle changes lane. There are several pedestrians."
                           "The ego vehicle turns left, avoiding parked cars.",
            },
            {
                "role": "user",
                "content": scene_caption,
            },
            
        ],
    )
    return gpt_response.choices[0].message.content