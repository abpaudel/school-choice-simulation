import numpy as np
from scipy.spatial import distance

def boston_mechanism(pref_i, pref_j, cap_j=None, name_ij=('I','J'), verbose=False):

    def top_nth_choice(n):
        pref = pref_i.argsort(1)
        return pref[:,n]
    def sort_by_priority(i_s, j):
        pj = pref_j[j]
        return sorted(i_s, key=lambda x:pj[x])

    def is_free(j):
        return np.sum(match[:,j]==1)<cap_j[j]

    if cap_j is None:
        cap_j = [1]*len(pref_j)
    match = np.zeros((len(pref_i), len(pref_j)))
    I, J = name_ij
    for step in range(len(pref_j)):
        if verbose:
            print(f'\nStep {step+1}')
        for j in range(len(pref_j)):
            i_s = np.where(top_nth_choice(step)==j)[0] # students whose step-th priority is j
            for i in sort_by_priority(i_s,j):
                if match[i].sum()==0: # if i is free
                    if verbose:
                        print(f'{I}{i} proposes to {J}{j}')
                    if is_free(j):
                        match[i,j] = 1
                        if verbose:
                            print(f'{I}{i} is matched with {J}{j}')
                    else:
                        if verbose:
                            print(f'{I}{i} is rejected because {J}{j} is not available')
    return match


def gale_shapley(pref_i, pref_j, cap_j=None, name_ij=('I','J'), verbose=False):

    def get_unadmitted():
        return np.where(match.max(1)==0)[0]

    def most_preferred(i):
        pref = pref_i[i].argsort()
        rejected = np.where(match[i]==-1)[0]
        available = [x for x in pref if x not in rejected]
        return available[0]

    def is_free(j):
        return np.sum(match[:,j]==1)<cap_j[j]

    def get_low_prefer(i, j):
        current = np.where(match[:,j]==1)[0]
        pref_curr = pref_j[j,current]
        if len(pref_curr)==0:
            return None
        max_pref = pref_curr.argmax()
        if pref_curr[max_pref]>pref_j[j,i]:
            return current[max_pref]
        return None

    if cap_j is None:
        cap_j = [1]*len(pref_j)
    match = np.zeros((len(pref_i), len(pref_j)))
    I, J = name_ij
    step = 0
    while 0 in match.max(1):
        step += 1
        if verbose:
            print(f'\nStep {step}')
        for i in get_unadmitted():
            j = most_preferred(i)
            if verbose:
                print(f'{I}{i} proposes to {J}{j}')
                print(f'{J}{j} is {"available" if is_free(j) else "not available"}')
            if is_free(j):  # if j is free, admit i to j
                match[i,j] = 1
                if verbose:
                    print(f'{I}{i} matched with {J}{j}')
                continue
            k = get_low_prefer(i,j) # k has lower pref than i but is admitted to j, so i can replace k
            if k is not None:
                match[i,j] = 1  # admit i to j
                match[k,j] = -1 # reject k from j
                if verbose:
                    print(f'{J}{j} prefers {I}{i} over current match {I}{k}')
                    print(f'{J}{j} leaves {I}{k}')
                    print(f'{I}{i} is matched with {J}{j}')
            else:
                match[i,j] = -1 # i is not admitted to j
                if verbose:
                    print(f'{J}{j} doesn\'t prefer {I}{i} over current match')
    return match


def feature_dist(n_std, n_sch):
    std_loc = np.random.rand(n_std,2)
    sch_loc = np.random.rand(n_sch,2)
    dist = distance.cdist(std_loc,sch_loc)
    return dist, std_loc, sch_loc


def feature_sibling(n_std, n_sch, p=0.5):
    sib_sch = np.zeros((n_std, n_sch))
    has_sib = np.random.binomial(1, p, n_std)
    sch_idx = np.random.randint(n_sch, size=n_std)
    sib_sch[np.arange(n_std),sch_idx] = has_sib
    return sib_sch, has_sib


def feature_school_tier(n_std, n_sch, p):
    t = np.random.choice(range(1,len(p)+1), n_sch, p=p)
    return np.tile(t, (n_std, 1)), t


def feature_gpa(n_std, n_sch, p):
    gpa = np.random.choice([2,3,4], n_std, p=p)
    return np.tile(gpa, (n_sch, 1)), gpa


def split_std(n_std, p_soph):
    n_soph = int(n_std*p_soph)
    idx_soph = np.random.choice(range(n_std), n_soph, replace=False)
    idx_sin = np.array(list(set(range(n_std))-set(idx_soph)), dtype=int)
    return idx_soph, idx_sin


def get_pref(n_std, n_sch):
    dist_std, std_loc, sch_loc = feature_dist(n_std, n_sch)
    sib_sch, has_sib = feature_sibling(n_std, n_sch, p=0.5)
    f_tiers, tier = feature_school_tier(n_std, n_sch, p=[0.1,0.2,0.3,0.4])
    f_rand = np.random.rand(n_std, n_sch)

    w_dist_std, w_sib, w_tier, w_rand = 0.5, 0.2, 0.2, 0.1
    w_dist_sch = 1

    pref_std = w_dist_std*dist_std/dist_std.max() + w_sib*(1-sib_sch) + w_tier*f_tiers/f_tiers.max() + w_rand*f_rand
    pref_sch = w_dist_sch*dist_std.T/dist_std.max()


    pr1 = (pref_sch>=0)&(pref_sch<0.3)
    pr2 = (pref_sch>=0.3)&(pref_sch<0.5)
    pr3 = (pref_sch>=0.5)&(pref_sch<0.7)
    pr4 = (pref_sch>=0.7)&(pref_sch<=1)

    pref_sch[pr1] = 1
    pref_sch[pr2] = 2
    pref_sch[pr3] = 3
    pref_sch[pr4] = 4

    return pref_std, pref_sch


def std_admit_choice(match, pref_std):
    '''Students are admitted to their nth priority school
    returns this n for each student'''
    admit = np.where(match==1)[1][:,None]
    pref_sort = pref_std.argsort(1)
    std_choice = np.argwhere(admit==pref_sort)[:,1]
    return std_choice


def alter_pref(pref_std, idx_std, swap_with=1):
    '''Strategy A: Swap 1st and 2nd if 1st is top school'''
    pref_new = pref_std.copy()
    pref_sort = pref_new.argsort(1)
    pr_1 = pref_sort[:,0]
    top_1 = np.bincount(pr_1).argmax()
    for i in idx_std:
        if pref_sort[i,0]==top_1:
            prf_1 = pref_sort[i,0]
            prf_2 = pref_sort[i,swap_with]
            v_prf_1 = pref_new[i,prf_1]
            v_prf_2 = pref_new[i,prf_2]
            pref_new[i,prf_1] = v_prf_2
            pref_new[i,prf_2] = v_prf_1
    return pref_new


def alter_pref2(pref_std, idx_std):
    '''Strategy B: Place 1st school to last if 1st is top school'''
    pref_new = pref_std.copy()
    pref_sort = pref_new.argsort(1)
    pr_1 = pref_sort[:,0]
    top_1 = np.bincount(pr_1).argmax()
    for i in idx_std:
        if pref_sort[i,0]==top_1:
            pref_new[i,top_1] = np.inf
    return pref_new


def alter_pref3(pref_std, idx_std):
    '''Strategy C: Find less popular school (i.e. mode of bottom half
    and put it on top if it is within your top half'''
    pref_new = pref_std.copy()
    pref_sort = pref_new.argsort(1)
    half = int(pref_std.shape[1]/2)
    btm_half = pref_sort[:,half:]
    unpop = np.bincount(btm_half.ravel()).argmax()
    for i in idx_std:
        if unpop in pref_sort[i,:half]:
            pref_new[i,unpop] = -np.inf
    return pref_new, unpop


def admit_to_top3(ch1, ch2):
    '''EM-Top3'''
    return np.sum((ch1>2)&(ch2+1<=2))


def admit_to_higher(ch1, ch2):
    '''EM-Higher'''
    return np.sum(ch1>(ch2+1))


def admit_to_unpop(match, idx_std, unpop):
    '''EM-Selected'''
    admit = np.where(match==1)[1][idx_std]
    return np.sum(admit==unpop)


def avg_rank_inc(ch1, ch2):
    return np.mean(ch1-(ch2+1))


def admit_to_second(ch1, ch2):
    return np.sum((ch1>1)&(ch2<=1))
