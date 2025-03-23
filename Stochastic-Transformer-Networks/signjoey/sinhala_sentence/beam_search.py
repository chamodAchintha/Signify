import torch
import torch.nn.functional as F
import numpy as np

def beam_search(
    model,
    encoder_output,
    src_mask,
    bos_index,
    eos_index,
    pad_index,
    beam_size=5,
    max_length=50,
    alpha=0.6,
    n_best=1,
):
    """
    Performs beam search decoding for a transformer-based model.
    
    :param model: Transformer-based decoder model
    :param encoder_output: Output from the encoder
    :param src_mask: Source mask from the input
    :param bos_index: Beginning-of-sequence token index
    :param eos_index: End-of-sequence token index
    :param pad_index: Padding token index
    :param beam_size: Number of beams to consider
    :param max_length: Maximum decoding length
    :param alpha: Length penalty factor
    :param n_best: Number of best hypotheses to return
    :return: (Final sequences, Scores)
    """
    batch_size = encoder_output.size(0)
    device = encoder_output.device
    
    # Initialize beams
    alive_seq = torch.full((batch_size * beam_size, 1), bos_index, dtype=torch.long, device=device)
    topk_log_probs = torch.zeros(batch_size, beam_size, device=device)
    topk_log_probs[:, 1:] = float('-inf')
    
    hypotheses = [[] for _ in range(batch_size)]
    results = {"predictions": [[] for _ in range(batch_size)], "scores": [[] for _ in range(batch_size)]}
    
    encoder_output = encoder_output.repeat_interleave(beam_size, dim=0)
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)
    
    for step in range(max_length):
        decoder_input = alive_seq
        logits = model.decode(encoder_output, src_mask, decoder_input)
        logits = logits[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs += topk_log_probs.view(-1, 1)
        curr_scores = log_probs.clone()
        
        if alpha > 0:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty
        
        curr_scores = curr_scores.view(batch_size, beam_size * model.vocab_size)
        topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
        
        if alpha > 0:
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()
        
        topk_beam_index = topk_ids // model.vocab_size
        topk_ids = topk_ids % model.vocab_size
        batch_index = (topk_beam_index + torch.arange(batch_size, device=device).view(-1, 1) * beam_size).view(-1)
        
        alive_seq = torch.cat([alive_seq.index_select(0, batch_index), topk_ids.view(-1, 1)], dim=-1)
        is_finished = topk_ids.eq(eos_index)
        
        if step + 1 == max_length:
            is_finished.fill_(True)
        end_condition = is_finished[:, 0].eq(True)
        
        if is_finished.any():
            predictions = alive_seq.view(batch_size, beam_size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                if end_condition[i]:
                    best_hyp = sorted(zip(topk_scores[i], predictions[i]), key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp[:n_best]):
                        results["scores"][i].append(score)
                        results["predictions"][i].append(pred)
            
            non_finished = end_condition.eq(False).nonzero().view(-1)
            if len(non_finished) == 0:
                break
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
    
    def pad_and_stack(hyps, pad_value):
        max_len = max(len(h) for h in hyps)
        return np.array([np.pad(h.cpu().numpy(), (0, max_len - len(h)), constant_values=pad_value) for h in hyps])
    
    final_outputs = pad_and_stack([r[0] for r in results["predictions"]], pad_value=pad_index)
    return final_outputs, results["scores"]
