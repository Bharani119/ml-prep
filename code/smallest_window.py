def smallest_window(s):
    if not s:
        return ""

    min_len = float("inf")
    min_window = ""

    for i in range(len(s)):
        for j in range(i + 3, len(s) + 1):
            window = s[i:j]
            if set("ABC").issubset(set(window)):
                if j - i < min_len:
                    min_len = j - i
                    min_window = window

    return min_window


s = "ABCHIKGAOLHBUHOCUHBHACJKGI"
print(smallest_window(s))
