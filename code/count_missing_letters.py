def count_missing_letters(word1, word2):
    len_diff = abs(len(word1) - len(word2))

    if len(word1) > len(word2):
        word1, word2 = word2, word1  # Swap words so that word1 is the shorter word

    missing_letters = len_diff + sum(
        1 for char1, char2 in zip(word1, word2) if char1 != char2
    )

    return missing_letters


# Input two words of different lengths
word1 = "simplify"
word2 = "sampled"

# Calculate the number of missing letters as replacements needed
missing_letters_count = count_missing_letters(word1, word2)

print(
    f"To transform '{word1}' into '{word2}', you need to replace {missing_letters_count} letters."
)
