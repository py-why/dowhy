#------------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets   #
# @Authors: Fredrik D. Johansson, Michael Oberst #
#------------------------------------------------#

import pandas as pd
import numpy as np

def compliance(D, R, inv_trans=lambda x,y : y):
    ops = {'<=': (lambda x,y : x <= y),
          '>': (lambda x,y : x > y),
          '>=': (lambda x,y : x >= y),
          '<': (lambda x,y : x < y),
          '==': (lambda x,y : x == y),
          '': (lambda x,y : x==True),
          'not': (lambda x,y : x==False)}

    Ws = []
    for r in R:
        W = []
        for c in r:
            try:
                v = float(c[2])
            except:
                v = c[2]
            W.append(ops[c[1]](inv_trans(c[0], D[c[0]].values), v))
        W = np.array(W)
        Ws.append(W)
    return Ws

def print_surgery_rules(R, D, labels, g_col, inv_trans=lambda x,y : y,
        log=lambda x: print(x)):

    prefix = lambda s,j : '  AND  ' + s if j>0 else '       ' + s
    lprefix = lambda s : ('          ' + s)[:55]
    lmax = np.max([len(lprefix(labels.get(c, c))) for c in D.columns.values])

    lpad = lambda s0,s1 : s0+' '*(lmax - len(s0)+2)+s1

    Cs = compliance(D, R, inv_trans)
    I0 = np.where(D[g_col].values==0)[0]
    I1 = np.where(D[g_col].values==1)[0]

    def conform_str(rule, literals):
        J = np.where(Cs[rule][literals,].prod(0) < 1)[0]
        mt0 = (1-D.iloc[J][g_col]).sum()
        mt1 = D.iloc[J][g_col].sum()

        m = 100*Cs[rule][literals,].prod(0).mean()
        n0 = Cs[rule][literals,][:,I0].prod(0).sum()
        n1 = Cs[rule][literals,][:,I1].prod(0).sum()
        m0 = 100*Cs[rule][literals,][:,I0].prod(0).mean()
        m1 = 100*Cs[rule][literals,][:,I1].prod(0).mean()
        return '%.1f%% (all), %d/%d, %.1f%%/%.1f%%, neg %d/%d' % (m, n0, n1, m0, m1, mt0, mt1)

    def print_rule(r, i_rule):
        data = {}
        j = 0
        for c in r:
            n = c[0]
            if n not in data:
                data[n] = {}

            data[n]['name'] = n
            if(not 'source' in data[n]):
                data[n]['source'] = []

            data[n]['source'].append(j)
            if c[1] in ['<=', '>', '>=', '<']:
                data[n]['type'] = 'range'
                if c[1] in ['<=', '<']:
                    data[n]['u'] = float(c[2])
                elif c[1] in ['>', '>=']:
                    data[n]['l'] = float(c[2])
            elif c[1] in ['not', '']:
                data[n]['type'] = 'binary'
                data[n]['v'] = not c[1] == 'not'
            elif c[1] in ['==']:
                data[n]['type'] = 'cat'
                data[n]['v'] = float(c[2])

            if 'u' not in data[n]:
                data[n]['u'] = np.nan
            elif 'l' not in data[n]:
                data[n]['l'] = np.nan
            j+=1

        D = pd.DataFrame([v for v in data.values()])

        j = 0
        # Male/female
        I_male = D[D['name']=='MALE'].index.values
        if(len(I_male)>0):
            if D.iloc[I_male[0]]['v']:
                log(lpad(prefix('MALE',j), conform_str(i_rule, np.hstack(D.iloc[I_male]['source']))))
            else:
                log(lpad(prefix('FEMALE',j), conform_str(i_rule, np.hstack(D.iloc[I_male]['source']))))
            j += 1

        # Continuous values
        def print_range_literal(d, j, i_literal, i_rule):
            if np.isnan(d['l']):
                s = '%s < %.1f' % (labels.get(d['name'], d['name']), d['u'])
            elif np.isnan(d['u']):
                s = '%s >= %.1f' % (labels.get(d['name'], d['name']), d['l'])
            else:
                s = '%s in [%.1f, %.1f]' % (labels.get(d['name'], d['name']), d['l'], d['u'])
            log(lpad(prefix(s,j), conform_str(i_rule, i_literal)))

        I_range = D[D['type']=='range'].index.values
        if(len(I_range)>0):
            for i in I_range:
                d = D.iloc[i]
                print_range_literal(d, j, np.hstack(D.iloc[i]['source']), i_rule)
                j += 1

        sets = [{'prefix': 'DX_', 'pos': 'HISTORY OF:', 'neg': 'NO HISTORY OF:'},
                {'prefix': 'SURGERY_', 'pos': 'SURGERY TYPE', 'neg': 'SURGERY TYPE NEITHER OF:'},
                {'prefix': 'THRDTDS_', 'pos': 'OPIOID TYPE', 'neg': 'OPIOID TYPE NEITHER OF:'},
                {'prefix': 'YEAR_', 'pos': 'YEAR AMONG:', 'neg': 'YEAR NOT AMONG:'}]

        # Sets of rules
        for s in sets:
            if 'v' not in D:
                D['v'] = False

            # Inclusion
            I_inc = D[(D['type']=='binary') & (D['v']==True) & (D['name'].str.find(s['prefix'])>=0)].index.values
            if(len(I_inc)>0):
                log(lpad(prefix(s['pos'], j), conform_str(i_rule, np.hstack(D.iloc[I_inc]['source']))))
                for i in I_inc: 
                    log(lpad(lprefix(labels.get(D.iloc[i]['name'], D.iloc[i]['name'])), conform_str(i_rule, np.hstack(D.iloc[i]['source']))))
                    j += 1

            # Exclusion
            I_exc = D[(D['type']=='binary') & (D['v']==False) & (D['name'].str.find(s['prefix'])>=0)].index.values
            if(len(I_exc)>0):
                log(lpad(prefix(s['neg'], j), conform_str(i_rule, np.hstack(D.iloc[I_exc]['source']))))

                for i in I_exc: 
                    log(lpad(lprefix(labels.get(D.iloc[i]['name'], D.iloc[i]['name'])), conform_str(i_rule, np.hstack(D.iloc[i]['source']))))
                    j += 1

        # Benzos
        I_ben = D[D['name']=='Benzodiazepine'].index.values
        if(len(I_ben)>0):
            log(lpad(prefix('Benzodiazepine == %.1f' % float(D.iloc[I_ben[0]]['v']),j), conform_str(i_rule, np.hstack(D.iloc[I_ben]['source']))))
            j += 1

    i = 0
    for r in R:
        s = 'EITHER RULE %d' % i
        if i>0:
            s = 'OR RULE %d' % i

        J = np.where(Cs[i].prod(0) < 1)[0]
        mt0 = (1-D.iloc[J][g_col]).sum()
        mt1 = D.iloc[J][g_col].sum()
        m = 100*Cs[i].prod(0).mean()
        n0 = Cs[i][:,I0].prod(0).sum()
        n1 = Cs[i][:,I1].prod(0).sum()
        m0 = 100*Cs[i][:,I0].prod(0).mean()
        m1 = 100*Cs[i][:,I1].prod(0).mean()
        log(lpad(s, '%.1f%% (all), %d/%d, %.1f%%/%.1f%%, neg %d/%d' % (m, n0, n1, m0, m1, mt0, mt1)))
        log('( ')
        print_rule(r, i)
        i+=1
    log(')')


def print_uti_rules(R, D, labels, g_col, inv_trans=lambda x,y : y):

    prefix = lambda s,j : '  AND  ' + s if j>0 else '       ' + s
    lprefix = lambda s : ('          ' + s)[:55]
    lmax = np.max([len(lprefix(labels.get(c, c))) for c in D.columns.values])

    lpad = lambda s0,s1 : s0+' '*(lmax - len(s0)+2)+s1

    Cs = compliance(D, R, inv_trans)
    I0 = np.where(D[g_col].values==0)[0]
    I1 = np.where(D[g_col].values==1)[0]

    def conform_str(rule, literals):
        J = np.where(Cs[rule][literals,].prod(0) < 1)[0]
        mt0 = (1-D.iloc[J][g_col]).sum()
        mt1 = D.iloc[J][g_col].sum()

        m = 100*Cs[rule][literals,].prod(0).mean()
        n0 = Cs[rule][literals,][:,I0].prod(0).sum()
        n1 = Cs[rule][literals,][:,I1].prod(0).sum()
        m0 = 100*Cs[rule][literals,][:,I0].prod(0).mean()
        m1 = 100*Cs[rule][literals,][:,I1].prod(0).mean()
        return '%.1f%% (all), %d/%d, %.1f%%/%.1f%%, neg %d/%d' % (m, n0, n1, m0, m1, mt0, mt1)

    def print_rule(r, i_rule):
        data = {}
        j = 0
        for c in r:
            n = c[0]
            if n not in data:
                data[n] = {}

            data[n]['name'] = n
            if(not 'source' in data[n]):
                data[n]['source'] = []

            data[n]['source'].append(j)
            if c[1] in ['<=', '>', '>=', '<']:
                data[n]['type'] = 'range'
                if c[1] in ['<=', '<']:
                    data[n]['u'] = float(c[2])
                elif c[1] in ['>', '>=']:
                    data[n]['l'] = float(c[2])
            elif c[1] in ['not', '']:
                data[n]['type'] = 'binary'
                data[n]['v'] = not c[1] == 'not'
            elif c[1] in ['==']:
                data[n]['type'] = 'cat'
                data[n]['v'] = float(c[2])

            if 'u' not in data[n]:
                data[n]['u'] = np.nan
            elif 'l' not in data[n]:
                data[n]['l'] = np.nan
            j+=1

        D = pd.DataFrame([v for v in data.values()])

        j = 0
        # Male/female
        I_male = D[D['name']=='MALE'].index.values
        if(len(I_male)>0):
            if D.iloc[I_male[0]]['v']:
                log(lpad(prefix('MALE',j), conform_str(i_rule, np.hstack(D.iloc[I_male]['source']))))
            else:
                log(lpad(prefix('FEMALE',j), conform_str(i_rule, np.hstack(D.iloc[I_male]['source']))))
            j += 1

        # Continuous values
        def print_range_literal(d, j, i_literal, i_rule):
            if np.isnan(d['l']):
                s = '%s < %.1f' % (labels.get(d['name'], d['name']), d['u'])
            elif np.isnan(d['u']):
                s = '%s >= %.1f' % (labels.get(d['name'], d['name']), d['l'])
            else:
                s = '%s in [%.1f, %.1f]' % (labels.get(d['name'], d['name']), d['l'], d['u'])
            log(lpad(prefix(s,j), conform_str(i_rule, i_literal)))

        I_range = D[D['type']=='range'].index.values
        if(len(I_range)>0):
            for i in I_range:
                d = D.iloc[i]
                print_range_literal(d, j, np.hstack(D.iloc[i]['source']), i_rule)
                j += 1

        sets = [{'prefix': 'DX_', 'pos': 'HISTORY OF:', 'neg': 'NO HISTORY OF:'},
                {'prefix': 'SURGERY_', 'pos': 'SURGERY TYPE', 'neg': 'SURGERY TYPE NEITHER OF:'},
                {'prefix': 'THRDTDS_', 'pos': 'OPIOID TYPE', 'neg': 'OPIOID TYPE NEITHER OF:'},
                {'prefix': 'YEAR_', 'pos': 'YEAR AMONG:', 'neg': 'YEAR NOT AMONG:'}]

        # Sets of rules
        for s in sets:
            if 'v' not in D:
                D['v'] = False

            # Inclusion
            I_inc = D[(D['type']=='binary') & (D['v']==True) & (D['name'].str.find(s['prefix'])>=0)].index.values
            if(len(I_inc)>0):
                log(lpad(prefix(s['pos'], j), conform_str(i_rule, np.hstack(D.iloc[I_inc]['source']))))
                for i in I_inc: 
                    log(lpad(lprefix(labels.get(D.iloc[i]['name'], D.iloc[i]['name'])), conform_str(i_rule, np.hstack(D.iloc[i]['source']))))
                    j += 1

            # Exclusion
            I_exc = D[(D['type']=='binary') & (D['v']==False) & (D['name'].str.find(s['prefix'])>=0)].index.values
            if(len(I_exc)>0):
                log(lpad(prefix(s['neg'], j), conform_str(i_rule, np.hstack(D.iloc[I_exc]['source']))))

                for i in I_exc: 
                    log(lpad(lprefix(labels.get(D.iloc[i]['name'], D.iloc[i]['name'])), conform_str(i_rule, np.hstack(D.iloc[i]['source']))))
                    j += 1

        # Benzos
        I_ben = D[D['name']=='Benzodiazepine'].index.values
        if(len(I_ben)>0):
            log(lpad(prefix('Benzodiazepine == %.1f' % float(D.iloc[I_ben[0]]['v']),j), conform_str(i_rule, np.hstack(D.iloc[I_ben]['source']))))
            j += 1

    i = 0
    for r in R:
        s = 'EITHER RULE %d' % i
        if i>0:
            s = 'OR RULE %d' % i

        J = np.where(Cs[i].prod(0) < 1)[0]
        mt0 = (1-D.iloc[J][g_col]).sum()
        mt1 = D.iloc[J][g_col].sum()
        m = 100*Cs[i].prod(0).mean()
        n0 = Cs[i][:,I0].prod(0).sum()
        n1 = Cs[i][:,I1].prod(0).sum()
        m0 = 100*Cs[i][:,I0].prod(0).mean()
        m1 = 100*Cs[i][:,I1].prod(0).mean()
        log(lpad(s, '%.1f%% (all), %d/%d, %.1f%%/%.1f%%, neg %d/%d' % (m, n0, n1, m0, m1, mt0, mt1)))
        log('( ')
        print_rule(r, i)
        i+=1
    log(')')
