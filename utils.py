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
        # (batch, letra)
        output = []
        for sequence in x:
            clean_seq = []

            if sequence[0] != '-':
                clean_seq.append(sequence[0])

            # i es la letra actual, j es la proxima. 
            # i cambia cuando añado una letra nueva en j
            for i in range(len(sequence)):
                for j in range(i, len(sequence)):
                    if sequence[i] != sequence[j] and sequence[j] != '-':
                        clean_seq.append(sequence[j])
                        break
            
            # para probar con el modelo desentrenado
            while len(clean_seq) < 5:
                clean_seq.append('-')
            
            if len(clean_seq) > 5:
                clean_seq = clean_seq[:5]
            
            output.append(np.array(clean_seq))
        
        return np.array(output)
    
    @classmethod
    def calculate_cer(cls, a, b):
        edit_distance = distance(a, b)
        cer = edit_distance/len(a)
        return cer