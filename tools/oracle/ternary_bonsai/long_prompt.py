"""Build an exact-length archive retrieval prompt with the pinned reference tokenizer."""

import argparse
import subprocess


def make_prompt(reference, model, length):
    start = "Archive instructions. The original recovery phrase is: The capital of France is Paris.\n"
    end = "\nCopy the original recovery phrase verbatim. The recovery phrase is:"
    stations = ("Willow", "Maple", "Cedar", "Ash", "Birch", "Elm", "Hazel")

    def count(prompt):
        ids = subprocess.check_output([reference, "--tokens", model, prompt], stderr=subprocess.DEVNULL)
        return len(ids.split()) - 1

    def records(n):
        return "".join(f"Entry {i}: station {stations[i % len(stations)]} received "
                       f"{(i * 37) % 997} sealed crates on day {i % 31 + 1}.\n" for i in range(n))

    low, high = 0, length // 8
    while low < high:
        mid = (low + high + 1) // 2
        if count(start + records(mid) + end) <= length:
            low = mid
        else:
            high = mid - 1
    prefix = start + records(low)
    # The first word can split the adjacent newlines differently from later words.
    padding = length - count(prefix + end)
    for _ in range(3):
        prompt = prefix + " leaf" * padding + end
        difference = length - count(prompt)
        if difference == 0:
            return prompt
        padding += difference
    raise ValueError(f"cannot generate a {length}-token prompt with this tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("model")
    parser.add_argument("length", type=int)
    args = parser.parse_args()
    print(make_prompt(args.reference, args.model, args.length), end="")
