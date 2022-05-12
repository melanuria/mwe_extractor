import json, sys, re, socket
from math import log
import os.path

homedir = os.path.expanduser("~")

def score_m2(model, astring):
    global a_re
    try:
        m = re.match(a_re, astring)
        r = m.group(1)
        a = m.group(2)
    except:
        print("no match astring: _{}_".format(astring))
        sys.exit(-1)
    ntokens = model["##tokens"]
    ntypes = len(model)

    if a in model:
        rdict = model[a]
        tokC_a = rdict['##tokens']
        typC_a = len(rdict)

        if r in rdict:
            tokC_ra = rdict[r]
        else:
            tokC_ra = 0
        p_a = (tokC_a + 1) / (ntokens + ntypes)
        p_ra = (tokC_ra + 1 ) / (tokC_a + typC_a)
        score = log(p_a) + log(p_ra)
    else:
        p_a = -log(ntokens + ntypes) #log(1 / (ntokens + ntypes))
        p_ra = log(model['##ratypes'] / ntokens)
        score = p_a + p_ra

    return score


def score_astrings(model, alist):
    slist = []
    for a in alist:
        score = score_m2(model[1], a)
        head, tail = [] , []
        for i in range(0, len(slist)):
            (sc, tmp) = slist[i]
            if sc < score:
                tail = slist[i:]
                break 
            else:
                head.append((sc,tmp))
        slist = head + [(score, a)] + tail
    return slist


a_re = re.compile(r'(<|[^<]+)(<.*)')

with open(homedir + '/TRmorph/lexicon/1M.m2', 'r', encoding='utf-8') as mfile:
    model = json.load(mfile)


def disambiguate(analyses):
    return [x[1] for x in score_astrings(model, analyses)]



