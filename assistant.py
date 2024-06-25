import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', choices=['cpu', 'gpu'])
parser.add_argument('--gputype', choices=['a100', 't4'])

def gpu_or_cpu(runtime):  # OR "cpu"
    
    """Set the runtime below according to whether you use GPU or CPU, to further place the model
    and the data to the correct place."""

    if runtime == "cpu":
        runtimeFlag = "cpu"
    elif runtime == "gpu":
        runtimeFlag = "cuda:0"
    else:
        print("Invalid runtime. Please set it to either 'cpu' or 'gpu'.")
        runtimeFlag = None

    return runtimeFlag

model_id = "Trelis/Llama-2-7b-chat-hf-function-calling-v2"


gputype = None
args = parser.parse_args()
runtime = args.runtime

if runtime == "gpu":
    
    """If GPU is used, then the model is loaded using 4-bit quantization.
    Also, check the float type according to your GPU."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if args.gputype == 'a100':
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
            cache_dir='',
    #         cache_dir=cache_dir, #in case you have enough space and want to reuse it
            torch_dtype=torch.bfloat16 #if on an A100 or Ampere architecture GPU,
            )
        
    elif args.gputype == 't4':
        
        model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        cache_dir='',
#         cache_dir=cache_dir, #in case you have enough space and want to reuse it
        torch_dtype=torch.float16, #if on a T4
        )
        
    else:
        sys.exit('Please specify the GPU type')
        
    # Not possible to use bits and bytes if using cpu only,
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=gpu_or_cpu(runtime), trust_remote_code=True)
    
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='', use_fast=True)

"""Here I define the 2 functions I need:"""

search_tickets_metadata = {
    "function": "search_tickets",
    "description": "Search tickets on aviasales.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
}

book_hotel_metadata = {
    "function": "book_hotel",
    "description": "Search for prices and available rooms for hotel.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
}

functionList = ''

functionList += json.dumps(search_tickets_metadata, indent=4, separators=(',', ': '))
functionList += json.dumps(book_hotel_metadata, indent=4, separators=(',', ': '))

def search_tickets(query):
    
    """Mock function for searching tickets."""

    return f"Searching aviasales for: {query}"

def book_hotel(query):
    
    """Mock function for booking hotels."""

    return f"Here are the available hotel options: {query}"

def generate(user_prompt):
    
    """This is a function to generate text based on user prompt."""

    B_INST, E_INST = "[INST] ", " [/INST]"
    B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"

    prompt = f"{B_FUNC}{functionList.strip()}{E_FUNC}{B_INST}{user_prompt.strip()}{E_INST}\n\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(gpu_or_cpu(runtime))

    outputs = model.generate(**inputs)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    """The decoder will return json together with the text output,
    which is either function or plain text. We cut out the json part via regex,
    and then, if the result is function, we return the output of the function,
    and if the result is plain text, we return the plain text itself.
    """
    
    result = re.search(r'\[/INST\](.*)', generated_text, re.DOTALL).group(1).strip()
    if "function" in result and "arguments" in result:

        function_call = json.loads(result)
        function_name = function_call.get("function")
        arguments = function_call.get("arguments", {})

        if function_name == "search_tickets":
            result = search_tickets(arguments['query'])

        elif function_name == "book_hotel":
            result = book_hotel(arguments['query'])

    return result
    
def main():
    
    print("""Hello. I am a virtual assistant and I am here to help you organize your travel.
    Feel free to ask anything you need. Print "Stop" to exit the chat.""")
    while True:
        user_input = input()
        if user_input.lower() == "stop":
            print("It was a pleasure to talk to you. Bye!")
            break
        print(generate(user_input))

if __name__ == "__main__":
    main()
    
