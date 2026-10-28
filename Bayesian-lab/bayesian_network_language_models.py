"""
AI Laboratory: Bayesian Networks and Autoregressive Language Models
Complete Implementation of First-Order and Second-Order Markov Language Models
Using Standard Python Data Structures and Random Sampling.
"""

import random
from collections import defaultdict, Counter

# Set random seed for reproducible sampling experiments
random.seed(42)

# ==============================================================================
# PART III: Build a Small Language Dataset
# ==============================================================================

RAW_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def preprocess_corpus(sentences, order=1):
    """
    Converts sentences to lowercase, tokenizes by whitespace,
    and prepends <START> token(s) and appends <END>.
    For a second-order model, we optionally use two <START> tokens or handle
    the initial pair (<START>, first_word) explicitly. Here we follow the lab's
    single <START> convention and handle initial transitions cleanly.
    """
    tokenized = []
    for sentence in sentences:
        words = sentence.strip().lower().split()
        tokens = ["<START>"] + words + ["<END>"]
        tokenized.append(tokens)
    return tokenized


# ==============================================================================
# PART IV - X: First-Order Autoregressive Language Model
# ==============================================================================

class FirstOrderLanguageModel:
    """
    First-Order Markov Language Model (Bayesian Network: X_1 -> X_2 -> ... -> X_T)
    Estimates P(X_t | X_{t-1}) from observed bigram transition counts.
    """

    def __init__(self):
        # Stores C(w_i, w_j): transition_counts[w_i][w_j] = count
        self.transition_counts = defaultdict(Counter)
        # Stores P(w_j | w_i): cpt[w_i][w_j] = probability
        self.cpt = {}
        self.vocabulary = set()

    def train(self, tokenized_sentences):
        """Counts consecutive token transitions and builds the CPT."""
        for tokens in tokenized_sentences:
            self.vocabulary.update(tokens)
            for i in range(len(tokens) - 1):
                w_prev = tokens[i]
                w_curr = tokens[i + 1]
                self.transition_counts[w_prev][w_curr] += 1

        # Construct conditional distribution P(X_t | X_{t-1})
        self.cpt = {}
        for w_prev, next_counts in self.transition_counts.items():
            total_outgoing = sum(next_counts.values())
            self.cpt[w_prev] = {
                w_curr: count / total_outgoing
                for w_curr, count in next_counts.items()
            }

    def get_distribution(self, prev_token):
        """Returns P(X_t | X_{t-1} = prev_token) as a dictionary."""
        return dict(self.cpt.get(prev_token, {}))

    def predict_most_probable(self, prev_token):
        """
        Computes argmax_w P(w | prev_token).
        If ties exist, returns the lexicographically first token among tied maxima,
        along with the full list of tied best tokens and the max probability.
        """
        dist = self.get_distribution(prev_token)
        if not dist:
            return None, [], 0.0
        max_prob = max(dist.values())
        best_tokens = sorted([w for w, p in dist.items() if abs(p - max_prob) < 1e-12])
        return best_tokens[0], best_tokens, max_prob

    def choose_next_token(self, prev_token, mode="sampling"):
        """
        Chooses next token according to:
        - mode='greedy': deterministic argmax_w P(w | prev_token)
        - mode='sampling': stochastic random draw from P(w | prev_token)
        Handles unseen context tokens gracefully by returning '<END>'.
        """
        dist = self.get_distribution(prev_token)
        if not dist:
            # Unobserved state fallback: safely terminate generation
            return "<END>"

        if mode == "greedy":
            best_token, _, _ = self.predict_most_probable(prev_token)
            return best_token
        elif mode == "sampling":
            words = list(dist.keys())
            probs = list(dist.values())
            return random.choices(words, weights=probs, k=1)[0]
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

    def generate_sentence(self, mode="sampling", max_length=25):
        """
        Generates a sentence starting after <START> until <END> or max_length.
        Includes loop protection for deterministic greedy mode if cycles occur.
        """
        current_token = "<START>"
        generated = []

        for _ in range(max_length):
            next_token = self.choose_next_token(current_token, mode=mode)
            if next_token == "<END>":
                break
            generated.append(next_token)
            current_token = next_token

        return " ".join(generated)

    def verify_normalization(self):
        """
        Part VII Test: Checks that sum_v P(v | w) == 1.0 for every conditioning word w.
        """
        results = {}
        all_valid = True
        for word, dist in sorted(self.cpt.items()):
            total = sum(dist.values())
            results[word] = total
            if abs(total - 1.0) > 1e-9:
                all_valid = False
        return all_valid, results

    def count_parameters_and_zeros(self):
        """
        Measures non-zero CPT parameters and zero-probability transitions
        across active conditioning states (excluding <END> as a predecessor)
        and possible target tokens (excluding <START> as a successor).
        """
        possible_sources = sorted(self.vocabulary - {"<END>"})
        possible_targets = sorted(self.vocabulary - {"<START>"})
        total_possible_cells = len(possible_sources) * len(possible_targets)
        nonzero_params = sum(len(dist) for dist in self.cpt.values())
        zero_transitions = total_possible_cells - nonzero_params
        return {
            "num_sources": len(possible_sources),
            "num_targets": len(possible_targets),
            "total_possible_cells": total_possible_cells,
            "nonzero_parameters": nonzero_params,
            "zero_transitions": zero_transitions,
        }


# ==============================================================================
# PART XI - XIII: Second-Order Autoregressive Language Model
# ==============================================================================

class SecondOrderLanguageModel:
    """
    Second-Order Markov Language Model (Bayesian Network: X_{t-2} -> X_t <- X_{t-1})
    Factorisation: P(X_1..X_T) = P(X_1) P(X_2|X_1) prod_{t=3}^T P(X_t | X_{t-2}, X_{t-1})
    With <START> prepended, X_1 follows <START> via P(X_1 | <START>), and subsequent
    tokens use pairs (X_{t-2}, X_{t-1}).
    """

    def __init__(self):
        # Stores initial transitions from <START>: C(<START>, w_1)
        self.initial_counts = Counter()
        self.initial_cpt = {}
        # Stores trigram counts C((w_{t-2}, w_{t-1}), w_t)
        self.trigram_counts = defaultdict(Counter)
        # Stores second-order CPT P(w_t | w_{t-2}, w_{t-1})
        self.cpt = {}
        self.vocabulary = set()

    def train(self, tokenized_sentences):
        """Counts observed triples (w_{t-2}, w_{t-1}, w_t) and constructs CPTs."""
        for tokens in tokenized_sentences:
            self.vocabulary.update(tokens)
            if len(tokens) >= 2:
                # Transition from <START> to first word
                self.initial_counts[tokens[1]] += 1
            for i in range(len(tokens) - 2):
                context = (tokens[i], tokens[i + 1])
                target = tokens[i + 2]
                self.trigram_counts[context][target] += 1

        # Normalize initial distribution P(X_1 | <START>)
        total_init = sum(self.initial_counts.values())
        if total_init > 0:
            self.initial_cpt = {
                w: c / total_init for w, c in self.initial_counts.items()
            }

        # Normalize second-order conditional distributions P(X_t | X_{t-2}, X_{t-1})
        self.cpt = {}
        for context, next_counts in self.trigram_counts.items():
            total_outgoing = sum(next_counts.values())
            self.cpt[context] = {
                w_curr: count / total_outgoing
                for w_curr, count in next_counts.items()
            }

    def get_distribution(self, context_pair):
        """Returns P(X_t | X_{t-2}, X_{t-1}) for a tuple (w_{t-2}, w_{t-1})."""
        return dict(self.cpt.get(context_pair, {}))

    def choose_next_token(self, context_pair, mode="sampling"):
        """Samples or greedily selects next token given (w_{t-2}, w_{t-1})."""
        dist = self.get_distribution(context_pair)
        if not dist:
            return "<END>"

        if mode == "greedy":
            max_prob = max(dist.values())
            best_tokens = sorted([w for w, p in dist.items() if abs(p - max_prob) < 1e-12])
            return best_tokens[0]
        elif mode == "sampling":
            words = list(dist.keys())
            probs = list(dist.values())
            return random.choices(words, weights=probs, k=1)[0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_sentence(self, mode="sampling", max_length=25):
        """Generates a sentence using the second-order Markov chain."""
        if not self.initial_cpt:
            return ""

        # Step 1: Sample first word after <START>
        if mode == "greedy":
            max_p = max(self.initial_cpt.values())
            first_word = sorted([w for w, p in self.initial_cpt.items() if abs(p - max_p) < 1e-12])[0]
        else:
            words = list(self.initial_cpt.keys())
            probs = list(self.initial_cpt.values())
            first_word = random.choices(words, weights=probs, k=1)[0]

        if first_word == "<END>":
            return ""

        generated = [first_word]
        context = ("<START>", first_word)

        # Step 2: Repeatedly sample X_t ~ P(X_t | X_{t-2}, X_{t-1})
        for _ in range(max_length - 1):
            next_token = self.choose_next_token(context, mode=mode)
            if next_token == "<END>":
                break
            generated.append(next_token)
            context = (context[1], next_token)

        return " ".join(generated)

    def verify_normalization(self):
        """Checks normalization for all observed bigram contexts."""
        results = {}
        all_valid = True
        for context, dist in sorted(self.cpt.items()):
            total = sum(dist.values())
            results[context] = total
            if abs(total - 1.0) > 1e-9:
                all_valid = False
        return all_valid, results

    def count_parameters_and_zeros(self):
        """
        Measures non-zero CPT parameters and zero-probability contexts/cells
        in the second-order model.
        """
        # Possible conditioning tokens (excluding <END>)
        active_tokens = sorted(self.vocabulary - {"<END>"})
        # Valid ordered pairs (w_{t-2}, w_{t-1}) where w_{t-1} != <START>
        possible_contexts = [
            (w1, w2)
            for w1 in active_tokens
            for w2 in sorted(self.vocabulary - {"<START>", "<END>"})
        ]
        possible_targets = sorted(self.vocabulary - {"<START>"})

        observed_contexts = len(self.cpt)
        unobserved_contexts = len(possible_contexts) - observed_contexts
        total_possible_cells = len(possible_contexts) * len(possible_targets)
        nonzero_params = sum(len(dist) for dist in self.cpt.values())
        zero_cells = total_possible_cells - nonzero_params

        return {
            "possible_contexts": len(possible_contexts),
            "observed_contexts": observed_contexts,
            "unobserved_zero_contexts": unobserved_contexts,
            "num_targets": len(possible_targets),
            "total_possible_cells": total_possible_cells,
            "nonzero_parameters": nonzero_params,
            "zero_cells": zero_cells,
        }


# ==============================================================================
# MAIN EXECUTION: Run All Laboratory Tasks & Print Deliverables
# ==============================================================================

def main():
    print("=" * 78)
    print("AI LABORATORY: BAYESIAN NETWORKS AND AUTOREGRESSIVE LANGUAGE MODELS")
    print("=" * 78)

    # 1. Preprocess Dataset (Part III)
    tokenized_corpus = preprocess_corpus(RAW_CORPUS)
    print("\n--- PART III: Tokenised Training Dataset ---")
    for idx, sent in enumerate(tokenized_corpus, 1):
        print(f"  Sentence {idx}: {' '.join(sent)}")

    # 2. Train First-Order Model (Part IV & V)
    model_1 = FirstOrderLanguageModel()
    model_1.train(tokenized_corpus)

    print("\n--- PART IV: First-Order Transition Counts & CPT ---")
    for prev_word in sorted(model_1.transition_counts.keys()):
        counts_str = ", ".join(
            f"{w}: {c}" for w, c in sorted(model_1.transition_counts[prev_word].items())
        )
        probs_str = ", ".join(
            f"{w}: {p:.4f}" for w, p in sorted(model_1.cpt[prev_word].items())
        )
        print(f"  Context '{prev_word:7s}' | Counts: {{{counts_str}}} | CPT: {{{probs_str}}}")

    # 3. Normalization Test (Part VII)
    print("\n--- PART VII: Probability Normalisation Verification (First-Order) ---")
    valid_1, norm_results_1 = model_1.verify_normalization()
    for word, total in norm_results_1.items():
        print(f"  sum_v P(v | '{word:7s}') = {total:.6f}")
    print(f"  All distributions sum to 1.0? {valid_1}")

    # 4. Next-Word Predictions (Part VIII)
    print("\n--- PART VIII: Next-Word Distribution & Argmax Predictions ---")
    query_words = ["<START>", "the", "cat", "dog", "sat", "ran", "on", "to"]
    for qw in query_words:
        dist = model_1.get_distribution(qw)
        best_w, tied_w, max_p = model_1.predict_most_probable(qw)
        tie_note = f" (tied: {tied_w})" if len(tied_w) > 1 else ""
        print(
            f"  X_t = '{qw:7s}' -> P(X_{{t+1}}|'{qw}') = {dist} "
            f"| argmax = '{best_w}' (p={max_p:.4f}){tie_note}"
        )

    # 5. Generate 20 Sentences (Part IX)
    print("\n--- PART IX: 20 Generated Sentences (First-Order Sampling) ---")
    generated_20 = []
    for i in range(1, 21):
        sent = model_1.generate_sentence(mode="sampling")
        generated_20.append(sent)
        print(f"  {i:2d}. {sent}")

    # 6. Deterministic vs Probabilistic Generation (Part X)
    print("\n--- PART X: Mode A (Greedy) vs Mode B (Sampling) - 5 Sentences Each ---")
    print("  Mode A (Greedy Generation - Always argmax_w P(w | w_prev)):")
    for i in range(1, 6):
        # Note: With greedy tie-breaking on 'the' -> 'cat' vs 'dog' (both 0.5)
        # or 'on' -> 'the' (1.0) -> 'cat', greedy mode hits a deterministic cycle
        # unless <END> is reached. We show exact greedy trace up to max_length=12:
        sent = model_1.generate_sentence(mode="greedy", max_length=12)
        print(f"    Greedy {i}: {sent}")

    print("  Mode B (Probabilistic Sampling - X_t ~ P(w | w_prev)):")
    for i in range(1, 6):
        sent = model_1.generate_sentence(mode="sampling")
        print(f"    Sample {i}: {sent}")

    # 7. Train Second-Order Model (Part XI & XII)
    model_2 = SecondOrderLanguageModel()
    model_2.train(tokenized_corpus)

    print("\n--- PART XI & XII: Second-Order Conditional Probability Table ---")
    print(f"  Initial P(X_1 | <START>): {model_2.initial_cpt}")
    for context in sorted(model_2.cpt.keys()):
        counts_str = ", ".join(
            f"{w}: {c}" for w, c in sorted(model_2.trigram_counts[context].items())
        )
        probs_str = ", ".join(
            f"{w}: {p:.4f}" for w, p in sorted(model_2.cpt[context].items())
        )
        ctx_label = f"({context[0]}, {context[1]})"
        print(f"  Context {ctx_label:18s} | Counts: {{{counts_str}}} | CPT: {{{probs_str}}}")

    print("\n--- PART XII: Probability Normalisation Verification (Second-Order) ---")
    valid_2, norm_results_2 = model_2.verify_normalization()
    for context, total in norm_results_2.items():
        ctx_label = f"({context[0]}, {context[1]})"
        print(f"  sum_v P(v | {ctx_label:18s}) = {total:.6f}")
    print(f"  All second-order distributions sum to 1.0? {valid_2}")

    print("\n--- PART XII: 20 Generated Sentences (Second-Order Sampling) ---")
    generated_2nd_20 = []
    for i in range(1, 21):
        sent = model_2.generate_sentence(mode="sampling")
        generated_2nd_20.append(sent)
        print(f"  {i:2d}. {sent}")

    # 8. Compare First-Order and Second-Order Models (Part XIII)
    print("\n--- PART XIII: Quantitative & Qualitative Model Comparison ---")
    stats_1 = model_1.count_parameters_and_zeros()
    stats_2 = model_2.count_parameters_and_zeros()
    print(f"  First-Order Model Stats : {stats_1}")
    print(f"  Second-Order Model Stats: {stats_2}")
    print(f"  Unique sentences in 20 samples (First-Order) : {len(set(generated_20))} -> {sorted(set(generated_20))}")
    print(f"  Unique sentences in 20 samples (Second-Order): {len(set(generated_2nd_20))} -> {sorted(set(generated_2nd_20))}")


if __name__ == "__main__":
    main()