text_str = "abcabcdbb"
long_str = ""
max_str = ""
for i in range(len(text_str)):
    long_str =text_str[i]
    for j in range(i+1,len(text_str)):
        if text_str[j] not in long_str:
            long_str+=text_str[j]
        else:
            break
    if len(long_str)>len(max_str):
        max_str = long_str
    # long_str = ""
print(max_str)
