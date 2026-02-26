import numpy as np
import torch
from Levenshtein import distance

class Utils():
    @classmethod
    def batch_w2i(cls, input, w2i):
        output = []
        lengths = []
        for word in input:
            int_label = [w2i[char] for char in word]
            output.append(int_label)
            lengths.append(len(word))
        return torch.tensor(output)
    
    @classmethod
    def transcription(cls, x, i2w):
        # (batch, letra)
        output = []
        for sequence in x:
            sub_output = []
            for letra in sequence:
                sub_output.append(i2w[str(letra.item())])

            output.append(sub_output)
        return np.array(output)

    @classmethod
    def clean(cls, x):
        # x (batch, pasos_de_tiempo)
        output = []
        for sequence in x:
            clean_seq = []
            last_char = None
            
            for char in sequence:
                if char != last_char:
                    if char != '-':
                        clean_seq.append(char)
                
                last_char = char

            if len(clean_seq) < 5:
                clean_seq.extend(['-'] * (5 - len(clean_seq)))
            else:
                clean_seq = clean_seq[:5]
                
            output.append(np.array(clean_seq))
            
        return np.array(output)
    
    @classmethod
    def calculate_cer(cls, a, b):
        edit_distance = distance(a, b)
        cer = edit_distance/len(a)
        return cer