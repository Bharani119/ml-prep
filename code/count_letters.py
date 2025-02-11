# RunLengthCoding Logic 
#“aagghjixyyyaa” -- > “2a2g1h1j1i1x3y2a” 

def count_let(x:str):
    curr_str = x[0]
    count_val = 0
    out_str = ""
    for i in x:
        if i in curr_str:
            count_val += 1
        elif i not in curr_str:
            out_str += str(count_val)+curr_str
            curr_str = i
            count_val = 1
            
    out_str += str(count_val)+curr_str
    return out_str

print(count_let("aagghjixyyyaa"))
