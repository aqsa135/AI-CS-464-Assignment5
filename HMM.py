# Aqsa Noreen

import random
import argparse
import codecs
import os
import numpy
import numpy as np


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this. DONE
    def load(self, basename):

        """reads HMM structure from transition (basename.trans),
                and emission (basename.emit) files,
                as well as the probabilities."""

        # dictionaries initiated
        transitions = {}
        emissions = {}

        # Transitions Read
        with open(f'{basename}.trans', 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    state_from = parts[0]
                    state_to = parts[1]
                    prob = float(parts[2])
                    if state_from not in transitions:
                        transitions[state_from] = {}
                    transitions[state_from][state_to] = prob

        # Read emissions
        with open(f'{basename}.emit', 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    state = parts[0]
                    observation = parts[1]
                    prob = float(parts[2])
                    if state not in emissions:
                        emissions[state] = {}
                    emissions[state][observation] = prob

        # Set the dictionaries to the HMM object
        self.transitions = transitions
        self.emissions = emissions
        # print(self.transitions)  # debug



   ## you do this. DONE
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""

        # Start with the initial state
        current_state = '#'
        states_sequence = []
        words_sequence = []

        for _ in range(n):
            # list of possible transitions and their probabilities
            transitions = self.transitions.get(current_state, {})
            states = list(transitions.keys())
            probabilities = list(transitions.values())
            current_state = np.random.choice(states, p=probabilities)
            states_sequence.append(current_state)

            # list of possible emissions and their probabilities
            emissions = self.emissions.get(current_state, {})
            observations = list(emissions.keys())
            probabilities = list(emissions.values())
            emitted_word = np.random.choice(observations, p=probabilities)
            words_sequence.append(emitted_word)

        return states_sequence, words_sequence

    #Part 3
    def forward(self, observations):
        """Run the forward algorithm to calculate the probability of the observation sequence."""
        num_observations = len(observations)
        states = list(self.transitions.keys())
        num_states = len(states)

        # Initialize forward probabilities matrix with 0s
        forward = np.zeros((num_observations, num_states))

        # Initialize first column of matrix
        for s in range(num_states):
            state = states[s]
            if state != '#':  # Skip the '#' state for emissions
                forward[0][s] = self.transitions['#'].get(state, 0) * self.emissions[state].get(observations[0], 0)

        # Compute forward probabilities
        for t in range(1, num_observations):
            for s in range(num_states):
                state = states[s]
                if state != '#':  # Skip the '#' state for emissions
                    for sp in range(num_states):
                        prev_state = states[sp]
                        if prev_state != '#':  # Skip the '#' state for transitions
                            forward[t][s] += forward[t - 1][sp] * self.transitions[prev_state].get(state, 0) * \
                                             self.emissions[state].get(observations[t], 0)

        # final probabilities Sum
        final_prob = np.sum(forward[num_observations - 1])

        return final_prob





    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.
    #Part 5

    def viterbi(self, observation):
        """given an observation,
                find and return the state sequence that generated
                the output sequence, using the Viterbi algorithm.
                """
        states = list(self.transitions.keys())
        states.remove('#')  # Remove the initial state as it does not emit and was giving trouble
        num_states = len(states)
        num_observations = len(observation)

        # epsilon to get rid of Log(0) warning
        epsilon = 1e-10

        # Initialize the Viterbi matrix with neg inf
        viterbi_matrix = np.full((num_states, num_observations), -np.inf)
        backpoint_matrix = np.zeros((num_states, num_observations), dtype=int)

        # Base case
        # fill in the first column of the Viterbi matrix
        for s, state in enumerate(states):
            if observation[0] in self.emissions[state]:
                viterbi_matrix[s][0] = np.log(self.transitions['#'][state]) + np.log(
                    self.emissions[state][observation[0]])

        # Recursion
        # fill in the rest of the Viterbi matrix
        for t in range(1, num_observations):
            for s, state in enumerate(states):
                for sp, prev_state in enumerate(states):
                    # Add epsilon to avoid log(0)
                    prob = (viterbi_matrix[sp][t - 1] +
                            np.log(self.transitions[prev_state].get(state, epsilon)) +
                            np.log(self.emissions[state].get(observation[t], epsilon)))
                    if prob > viterbi_matrix[s][t]:
                        viterbi_matrix[s][t] = prob
                        backpoint_matrix[s][t] = sp

        # Termination
        # find the most probable last state
        best_state = np.argmax(viterbi_matrix[:, num_observations - 1])
        best_path_prob = viterbi_matrix[best_state, num_observations - 1]

        #  backtracking path
        best_path = [states[best_state]]
        for t in range(num_observations - 1, 0, -1):
            best_state = backpoint_matrix[best_state][t]
            best_path.insert(0, states[best_state])

        return best_path, best_path_prob



def main():

    #part 1
    model = HMM()
    model.load('two_english')

    # Print the entire transitions dictionary to check its structure
    print("Transitions dictionary:")
    print(model.transitions)

    # Print the entire emissions dictionary to check its structure
    print("\nEmissions dictionary:")
    print(model.emissions)

    # print a specific entry from each dictionary to check the format
    print("\nSample transition probabilities for '#' state:")
    if '#' in model.transitions:
        print(model.transitions['#'])
    else:
        print("No '#' state found in transitions.")

    print("\nSample emission probabilities for 'C' state:")
    if 'C' in model.emissions:
        print(model.emissions['C'])
    else:
        print("No 'C' state found in emissions.")

    #
    #
    # # model = HMM()
    # #part2
    model.load('partofspeech.browntags.trained')

    # Generate 20 random observations (tags and words)
    tags, words = model.generate(20)
    print('Tags:', ' '.join(tags))
    print('Words:', ' '.join(words))

    # Part 3

    parser = argparse.ArgumentParser()
    parser.add_argument('basename', help='Base name for the .trans and .emit files')
    parser.add_argument('--forward', help='File with the sequence of observations for the forward algorithm')
    args = parser.parse_args()

    model = HMM()
    model.load(args.basename)

    # Debugging Print the transitions dictionary after loading
    # print("Transitions dictionary after loading:")
    # print(model.transitions)

    if args.forward:
        with open(args.forward, 'r') as f:
            observations = f.read().strip().split()

            # Debugging: Print the transitions dictionary right before calling forward
            # print("Transitions dictionary right before forward:")
            # print(model.transitions)

            final_state_prob = model.forward(observations)
            print(f'The probability of the final state is: {final_state_prob}')
        # final_state_prob = model.forward(observations)
        # print(f'The probability of the final state is: {final_state_prob}')


    #Part 4
    # model = HMM()
    model.load('partofspeech.browntags.trained')

    with open('ambiguous_sents.obs', 'r') as f:
        observations = f.read().strip().split()

    best_path, best_path_prob = model.viterbi(observations)
    print('Best path:', ' '.join(best_path))
    print('Probability of the best path:', best_path_prob)


if __name__ == "__main__":
    main()




