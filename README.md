# Virtual assistant

In this project I create a virtual assistant for searching for flight tickets and booking hotels.
One way to implement the searching into generative network is so called function calling.
Since the task required using open-source model with not more than 7B parameters, I chose [llama-2].
Llama 2 is an auto-regressive language model that uses an optimized transformer architecture.
To add the capability of function calling, the model needs to be fine-tuned in such a way that, getting a certain user prompt,
it returns a structured json object that contains SOMETHING

I did not fine-tune the model myself since I had limited free colab resources. Instead I used [this] model, which is already fine-tuned
for function calling. It requires creating a string called `functionList` that contains list of functions with the following requisites:
function name, description and the desired arguments.
I put two functions to the list: the one for searching tickets and another one for booking hotels.
I process output in such a way that if the output is a function, I return the output of a function, and if the output is plain text,
I return the text itself.
The result looks like this:

<img src="https://github.com/AnnaLebedeva/mts/blob/main/output.png" alt="drawing" width="700"/>

[llama-2]: <https://arxiv.org/abs/2307.09288>
[this]: <https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2>

## How to use:

Clone the repository:

    git clone https://github.com/AnnaLebedeva/mts.git

Go to folder and install the necessary dependencies:

    cd mts
    pip install -r requirements.txt

Run the assistant.py file. Please specify which devide you use by setting `--runtime` option to `gpu` or `cpu`.
If you use gpu, please set the option `--gputype` for either `a100` or `t4`(for colab).
Example of command if you use gpu a100:

    python assistant.py --runtime gpu --gputype a100

