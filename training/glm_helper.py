import numpy as np
from external.progen3.src.progen3.batch_preparer import prepare_glm_string_from_spans

# For infilling task - following appendix C infilling details in Progen3 paper
def generate_glm_instance(sequence):
    rng = np.random.default_rng()

    L_total = len(sequence)

    # 1. Maximum length fraction to infill for this sequence
    f_options = [0.15, 0.25, 0.5, 0.8]
    f_probs = [0.28, 0.3, 0.28, 0.14]
    f = rng.choice(f_options, p=f_probs)
    max_total_len = L_total * f

    # 2. Sample lengths using Mixture of Gaussians
    gaussians = [(10, 5), (30, 10), (70, 20), (200, 50), (400, 100)]
    sampled_lengths = []
    current_total = 0

    while current_total < max_total_len:
        mu, sigma = gaussians[rng.integers(len(gaussians))]
        # Ensure that chosen length > 0
        while True:
            l = int(rng.normal(mu, sigma))
            if l > 0:
                break
        
        if current_total + l > max_total_len:
            break

        sampled_lengths.append(l)
        current_total += l

    if not sampled_lengths:
        return sequence

    # 3. Weighted placement into free intervals
    free_intervals = [(0, L_total)]
    spans = {}

    def place_span(length, free_intervals):
        # Build list of intervals with enough room
        valid = []
        for a, b in free_intervals:
            count = b - a - length + 1
            if count > 0:
                valid.append((a, b, count))

        if not valid:
            return None
        
        weights = [v[2] for v in valid]
        total = sum(weights)
        r = rng.integers(total)
        cumulative = 0
        for a, b, count in valid:
            cumulative += count
            if r < cumulative:
                start = rng.integers(a, b - length + 1)
                return start, start + length
            
    def update_intervals(free_intervals, start, end):
        new_intervals = []
        for a, b in free_intervals:
            if end <= a or start >= b:
                new_intervals.append((a, b))
            else:
                if a < start:
                    new_intervals.append((a, start))
                if end < b:
                    new_intervals.append((end, b))
        return new_intervals
    
    rng.shuffle(sampled_lengths)

    for length in sampled_lengths:
        span = place_span(length, free_intervals)
        if span is None:
            break
        start, end = span
        spans[(start, end)] = length
        free_intervals = update_intervals(free_intervals, start, end)

    if not spans:
        return sequence
    
    # 3. Format into string for batch preparer
    spans = dict(sorted(spans.items()))
    glm_suffix = prepare_glm_string_from_spans(spans)
    return f"{sequence}{glm_suffix}"