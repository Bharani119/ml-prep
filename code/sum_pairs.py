lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

res = 10

def identify_sum_of_pair_eq_res(res_num, list_numbers):
    pairs_with_res = []
    for i in range(len(list_numbers)):
        for j in range(i+1,len(list_numbers)):
            if list_numbers[i] + list_numbers[j] == res_num:
                pairs_with_res.append((list_numbers[i],list_numbers[j]))
    return pairs_with_res
        

print(identify_sum_of_pair_eq_res(res, lst))

from itertools import combinations,permutations

print([i for i in combinations(lst,2)])
# print([i for i in permutations(lst,2)])

outli = []
for comb in [i for i in combinations(lst,2)]:
    if sum(comb) == res:
        outli.append(comb)

print(outli)