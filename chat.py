import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained data from a PyTorch model file
FILE = "data.pth"
data = torch.load(FILE)

# Extract information from the loaded data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load the NeuralNet model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Set the bot's name
bot_name = "Sam"

# Start a chat loop
print("Let's chat! (type 'quit' to exit)")
while True:
    # Get user input
    sentence = input("You: ")

    # Check if the user wants to quit
    if sentence == "quit":
        break

    # Tokenize and process the user input
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Make a prediction using the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Determine the response based on the model's prediction
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
