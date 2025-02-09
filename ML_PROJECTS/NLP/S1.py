import torch
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive, negative):
        """
        Preprocesses the input text data by encoding words into integers and 
        padding the sequences to make them of uniform length.

        Args:
            positive (list of str): List of positive sentences.
            negative (list of str): List of negative sentences.

        Returns:
            torch.Tensor: A padded tensor where each row corresponds to an encoded sentence.
        """

        # Combine positive and negative sentences into a single list
        combined = positive + negative

        # Create a set of all unique words from the combined sentences
        words = set()
        for sentence in combined:
            for word in sentence.split():  # Split each sentence into words
                words.add(word)

        # Sort the unique words to ensure consistent word-to-integer mapping
        sorted_list = sorted(list(words))

        # Create a dictionary to map words to unique integers
        word_to_int = {}
        for i, word in enumerate(sorted_list):
            word_to_int[word] = i + 1  # Start integer indices from 1

        # Initialize a list to store encoded sequences (unpadded)
        unpadded = []

        # Encode each sentence into a sequence of integers
        for sentence in combined:
            encoded = []
            for word in sentence.split():
                encoded.append(word_to_int[word])  # Map word to its integer representation
            unpadded.append(torch.tensor(encoded))  # Convert the sequence to a tensor

        # Pad all sequences to the same length and return the resulting tensor
        return pad_sequence(unpadded, batch_first=True)  # Pad sequences with zeros

# Example Usage
if __name__ == "__main__":
    # Instantiate the Solution class
    sol = Solution()

    # Example positive and negative sentences
    positive = ["I love programming", "AI is fascinating"]
    negative = ["I dislike bugs", "Debugging is tough"]

    # Generate the padded dataset
    padded_dataset = sol.get_dataset(positive, negative)

    # Print the resulting padded tensor
    print(padded_dataset)
        
        