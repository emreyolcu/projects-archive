import torch
import torch.nn.functional as F

import utils


class BeamNode:
    def __init__(self, hypothesis, logprob, alphas):
        self.hypothesis = hypothesis
        self.logprob = logprob
        self.alphas = alphas


def beam_search(model, src_sent, beam_size=5, max_decoding_time_step=70):
    tgt_vocab_len = len(model.vocab.tgt)
    src_encodings, encoder_cell = model.encoder([src_sent])
    src_encodings = src_encodings.permute(1, 0, 2)
    src_attn_scores = model.decoder.attention.proj_encoder(src_encodings)

    prev_context = torch.zeros(src_encodings.shape[0], model.encoder.hidden_size).to(utils.DEVICE)
    prev_state = model.decoder.init_state(encoder_cell)

    beam = [BeamNode([model.vocab.tgt['<s>']], 0, [])]
    done = []

    for i in range(max_decoding_time_step):
        if len(done) == beam_size:
            break

        score, context, state, alpha = model.decoder.decode_one_word(
            model.decoder.embedding(torch.tensor([node.hypothesis[-1] for node in beam], dtype=torch.long).to(utils.DEVICE)),
            prev_context,
            prev_state,
            src_encodings.expand(len(beam), src_encodings.size(1), src_encodings.size(2)),
            src_attn_scores.expand(len(beam), src_attn_scores.size(1), src_attn_scores.size(2)),
        )

        prev_logprobs = torch.tensor([node.logprob for node in beam], dtype=torch.float).to(utils.DEVICE)
        logprobs = (F.log_softmax(score, dim=1) + prev_logprobs.unsqueeze(1)).flatten()
        indices = torch.topk(logprobs, k=(beam_size - len(done)))[1]

        next_beam = []
        active = []

        for index in indices.cpu().numpy():
            parent_index, vocab_index = index // tgt_vocab_len, index % tgt_vocab_len
            node = BeamNode(beam[parent_index].hypothesis + [vocab_index], logprobs[index],
                            beam[parent_index].alphas + [alpha[parent_index]])
            if vocab_index == model.vocab.tgt['</s>']:
                done.append(node)
            else:
                next_beam.append(node)
                active.append(parent_index)

        active = torch.tensor(active, dtype=torch.long).to(utils.DEVICE)
        prev_context = context[active]
        prev_state = (state[0][active], state[1][active])
        beam = next_beam

    if not done:
        done = beam

    sorted_done = sorted(
        [(list(map(lambda x: model.vocab.tgt.id2word[x], node.hypothesis)), node.logprob.item(), node.alphas) for node in done],
        key=lambda x: x[1] + 0.2 * len(x[0]),
        reverse=True,
    )

    return [x[0] for x in sorted_done], [x[2] for x in sorted_done]
