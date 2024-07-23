import json
import numpy as np

from difflib import get_close_matches
from keras.models import load_model,Model
from keras.layers import Input
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
import re




class ChatBot:
    def __init__(self):
        # Load the seq2seq model
        self.training_model = load_model('training_model.h5')
        encoder_inputs = self.training_model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = self.training_model.layers[2].output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        latent_dim = 256
        decoder_state_input_hidden = Input(shape=(latent_dim,))
        decoder_state_input_cell = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
        #Decoder
        dimensionality = 256 
        num_decoder_tokens = 3101
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
        decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Assuming that you have the decoder_lstm and decoder_dense layers defined somewhere
        decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_hidden, state_cell]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        # Load the knowledge base
        self.knowledge_base = self.load_knowledge_base('knowledge_base.json')
        self.question_dict = self.build_question_dict(self.knowledge_base["questions"])

        self.negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
        self.exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    #this function reads a JSON file containing the knowledge base, converts it into a dictionary, and returns that dictionary.
    def load_knowledge_base(self, file_path: str) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            data: dict = json.load(file)
        return data

    #this function transforms a list of question-answer pairs into a dictionary, where each question is associated with its corresponding answer.
    def build_question_dict(self, questions: list) -> dict:
        return {q["Question"]: q["Answer"] for q in questions}

    #his function takes a dictionary representing the knowledge base and saves it to a JSON file
    def save_knowledge_base(self, file_path: str, data: dict):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

    #this function uses a fuzzy matching approach to find the best match for the user's question in the existing question-answer pairs. 
    def find_best_match(self, user_question: str) -> str | None:
        matches: list = get_close_matches(user_question, self.question_dict.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    #this function provides a convenient way to get the answer for a given question from the existing question-answer pairs stored in the question_dict dictionary.
    def get_answer_for_question(self, question: str) -> str | None:
        return self.question_dict.get(question)

    #this function prepares the user's input by converting it into a format that can be fed into a neural network, where each token is represented by a binary vector in the matrix.
    def string_to_matrix(self, user_input):
        num_encoder_tokens = 282
        max_encoder_seq_length = 53
        # Assume that num_encoder_tokens and input_features_dict are defined somewhere
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32'
        )
        with open('input_features_dict','r') as json_file:
            input_features_dict = json.load(json_file)
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    #this function uses the trained encoder and decoder models to generate a response by iteratively predicting tokens until the stop condition is met, and it returns the decoded sentence.
    def decode_response(self, test_input):
        num_decoder_tokens = 3101
        max_decoder_seq_length = 1642
        with open('target_features_dict','r') as json_file:
            target_features_dict = json.load(json_file)
        states_value = self.encoder_model.predict(test_input)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_features_dict['<START>']] = 1.
        with open('reverse_target_features_dict','r') as json_file:
            reverse_target_features_dict = json.load(json_file)
        decoded_sentence = ''
        stop_condition = False

        while not stop_condition:
            output_tokens, hidden_state, cell_state = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [hidden_state, cell_state]

        return decoded_sentence

    #this function initiates a chat by prompting the user for input, checks if the user's response indicates a desire to end the conversation, and if not, passes the user's response to the chat method for further interaction.
    def start_chat(self):
        user_response = input("Hi, I'm a chatbot trained on random dialogs. AMA!\n")

        if user_response in self.negative_responses:
            print("Ok, have a great day!")
            return

        self.chat(user_response)

    #this function handles the ongoing chat interaction in a loop, repeatedly generating responses based on the user's input and prompting the user for further input until the user decides to exit the chat.
    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply) + "\n")

    #this function takes the user's input, converts it into a matrix, decodes the model's response, and then cleans up the response by removing any special tokens before returning the final chatbot-generated response.
    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = self.decode_response(input_matrix)
        chatbot_response = chatbot_response.replace("<START>", '')
        chatbot_response = chatbot_response.replace("<END>", '')
        return chatbot_response

    #this function checks if any exit commands are present in the user's input.
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
        return False

    #his function runs an interactive chat loop where the chatbot responds to user input based on pre-existing knowledge or learns new information from the user.
    def run_chat_bot(self):
        while True:
            user_input: str = input('Ask me something: (type quit to exit) ')
            if user_input == 'quit':
                print("Bye! Have a great day ahead")
                break

            best_match: str | None = self.find_best_match(user_input)

            if best_match:
                answer: str = self.get_answer_for_question(best_match)
                print(f'Bot: {answer}')
            else:
                print('Bot: I don\'t know the answer. You can rephrase your prompt or you can teach me? ')
                new_answer: str = input('Type the answer or "skip" to enter a new prompt: ')
                if new_answer.lower() != 'skip':
                    self.knowledge_base["questions"].append({"Question": user_input, "Answer": new_answer})
                    self.question_dict = self.build_question_dict(self.knowledge_base["questions"])
                    self.save_knowledge_base('knowledge_base.json', self.knowledge_base)
                    print('Bot: Thank you! I learned something today')

#when the script is executed directly (not imported as a module), it creates an instance of the ChatBot class and runs the chatbot loop. 
if __name__ == '__main__':
    ch = ChatBot()
    ch.run_chat_bot()