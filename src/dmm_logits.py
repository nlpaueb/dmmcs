from dmm import DMM
from transformers import LogitsProcessor
import torch

from utils.utilities import (
  normalize_dmm,
  normalize_lm
)

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

  

  def __call__(self, input_ids, scores):
    # for every beam (partially generated sentence)
    hist_conf = list()
    final_scores = list()
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
        
        running_caption = self.tokenizer.decode(beam_input_ids)
        hist_conf.append(self.dmm.dmm_handler(running_caption, self.tags))
    
    final_scores = list()
    
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):

        running_caption = self.tokenizer.decode(beam_input_ids)

        dmm_loss = self.dmm.dmm_handler(running_caption, self.tags)

        dmm_component = normalize_dmm((self.alpha * dmm_loss))

        lm_component = normalize_lm((1-self.alpha) * (-1*beam_scores))

        scores[beam_index] = -1 * (dmm_component + lm_component)
    
    return scores
