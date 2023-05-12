
import torch

class Beamsearch():
    '''
    Input: Model's logits 

    Step 1: Innitialize the beams with emty sequence, and probability of 0
    Step 2: Find the first next token based on model's outputs (logits), add the token to sequence
            and add its probability (obtained by taking the softmax) to the beams.
    Step 3: Compute the probability of updated beam sequence by taking the product p_t and p_t-1 
            (i.e. P(y|t-1...t) = P(y|t-1) *...* P(y|t))
            Apply the log(probability) to avoid vanishing issue
    Step 4: Select the top sequences with top k probabilities. Append new seqs and probs to the beam
    Step 5: Repeat steps 3 and 4 until beam sequence length reaches the max sequence length. 

    '''
    def __init__(self, model_outputs):
        self.model_outputs = model_outputs

    def beam_search(self, k, max_len):
        i=1
        # innitialize the beams with empty sequence prob and 0 score
        beams = [([torch.tensor([0])], 0)]
        # get probabilities of the 1st token
        current_logit = self.model_outputs[i,:]
        current_prob = torch.log_softmax(current_logit, dim=-1) + beams[0][1]
        while i < max_len:
            new_beams = []
            for beam in beams:  
                pre_probs =  current_prob 
                # get the top k token probabilities to add to the beam   
                topk_probs, topk_indices = torch.topk(pre_probs, k) 
                # concat the new sequence to current sequence with probability
                for s in range(k):
                    new_seq = torch.cat([beam[0], topk_indices[0][s]]) 
                    new_prob = topk_probs[0][s]
                    new_beams.append((new_seq, new_prob))
                # compute the probabilities of current k beam sequences
                current_prob = torch.matmul(torch.log_softmax(current_logit, dim=-1), beam[1])
            beams = new_beams
            i+=1

        return beams


