from dmm import DMM
from transformers import LogitsProcessor
import torch

class DMMLogits(LogitsProcessor):
  def __init__(self, dmm, tags, alpha, tokenizer):
    """
    vocab is a dictionary where the keys are tokens
    and the values are the corresponding ids.
    """
    self.dmm = dmm
    self.tags = tags
    self.alpha = alpha
    self.tokenizer = tokenizer

  def normalize_dmm(self, value):
    xmin = 0
    xmax = 1.5

    norm = ((value - xmin) / (xmax-xmin))

    return norm

  def normalize_lm(self, tensor_):
    xmin = 2
    xmax = 22

    norm = ((tensor_ - xmin) / (xmax-xmin))

    return norm

  def __call__(self, input_ids, scores):
    # for every beam (partially generated sentence)
    hist_conf = list()
    final_scores = list()
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
        
        running_caption = self.tokenizer.decode(beam_input_ids)
        hist_conf.append(self.dmm.dmm_handler(running_caption, self.tags, calc_dmm_loss=False))
    
    final_scores = list()
    
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):

        running_caption = self.tokenizer.decode(beam_input_ids)

        dmm_loss = self.dmm.dmm_handler(running_caption, self.tags, calc_dmm_loss=True)

        # dmm_component = (self.alpha * (1-hist_conf[beam_index]) * dmm_loss) --> previous form of the dmmcs type
        dmm_component = (self.alpha * dmm_loss)

        dmm_component = self.normalize_dmm(dmm_component)

        # snt_component = (1-self.alpha) * hist_conf[beam_index] * (-1*beam_scores) --> previous form of the dmmcs type
        lm_component = (1-self.alpha) * (-1*beam_scores)

        lm _component = self.normalize_lm(lm_component)
        print('------------------------------------------') # for visualization purposes

        final_scores_tensor = -1 * (dmm_component + lm_component)

        scores[beam_index] = final_scores_tensor
    
    return scores
