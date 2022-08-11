#The ledgend for the generated graphs
from graphviz import Digraph
import copy
import pandas as pd
import numpy as np

AvaOrangeLight = "#FF5800"
AvaOrangeDark  = "#DC4600"
AvaGoldLight = "#FFB414"
AvaGoldDark  = "#E6A61C"

l = Digraph('Ledgend',format='png')
l.attr(overlap='scale')
l.attr(splines='True')
l.attr('edge', fontsize='12')

l.attr('node',fillcolor="white")
l.attr('node',color='black')
l.attr('node',shape='')
l.attr('node',style="filled")
l.attr('node',fontsize='12') 

l.attr('node',shape='triangle')   
l.node("Exogenous")

l.attr('node',shape='box')
l.node("Observed")

l.attr('node',shape='circle')
l.node("Latent")

l.attr('node',shape='invtriangle')
l.node("Outcome")

l.attr('node',shape='plain')
l.attr('node',fillcolor="white")
l.attr('node',color="white")
l.node("intercept")

l.edge("Exogenous" , "Observed"  , label="Effect, adjusted pvalue < 0.05"                        , color="black"                  )
l.edge("Observed"  , "Latent"    , label="Effect, adjusted pvalue > 0.05"                        , color=AvaOrangeLight           )
l.edge("Exogenous" , "Observed"  , label="Covariance Equivalance, adjusted pvalue < 0.05"        , color="black"      , dir="both")
l.edge("Observed"  , "Latent"    , label="Covariance Equivalance, adjusted pvalue > 0.05"        , color=AvaGoldLight , dir="both")
l.edge("Latent" , "Outcome"      , label= ""                                                     , color="black"                  )

l.save()
l.render(view=True)
display(l)


def scm_results_plot(dataVars, scmedges_in, name):
    ''' Good references
        https://graphviz.org/doc/info/shapes.html
        https://graphviz.readthedocs.io/en/stable/examples.html
        https://graphviz.org/doc/info/shapes.html
        https://stackoverflow.com/questions/66092429/how-to-format-edge-labels-in-graphviz

        Graph conventions:
        https://cjvanlissa.github.io/tidySEM/articles/sem_graph.html

        - Nodes
          - Exhogenous (no parents)             : Triangle
          - Variable (with parents and children): Square [box]
          - Latent variable (not in the data)   : Circle
          - Outcomes (no children)              : Inverted triangle
        - Edges
          - Causal/Regression efects: straingt arrow
          - Covariance: dashed curve
          - (Residual) variance: double headed arrow

        Param: dataVars - list of variables in the dataset
        Param: scmedges_in - Pandas dataframe similer to sempoy output
        Param: name - text to beused to name graph and resulting <name>.png
        '''
    from graphviz import Digraph
    import copy
    import pandas as pd
    import numpy as np

    AvaOrangeLight = "#FF5800"
    AvaOrangeDark  = "#DC4600"
    AvaGoldLight = "#FFB414"
    AvaGoldDark  = "#E6A61C"
   
    scmdges = copy.copy(scmedges_in) # legacy from prev code
    scmdges=pd.DataFrame(scmdges[(scmdges['op'] == "~")])
    #name intercepts explicityl
    
    scmdges['intercept']= (scmdges['rval']=="Intercept")
    scmdges.loc[scmdges['intercept'],'rval'] = scmdges['lval']+" Intercept"
    
    interceptVarsDS = scmdges[scmdges['intercept']]
    interceptVars = interceptVarsDS['rval']
    interceptsCount = len(interceptVars)
    
    #idenfitfy exogenous varialbe = no incomming arrows
    if interceptsCount>0:
       trS=scmdges[scmdges['intercept']==False]
       tr=pd.DataFrame(trS['rval'])
    else:   
       tr=pd.DataFrame(scmdges['rval'])
    r=tr.drop_duplicates()
    r.columns=["f"]
    tl=pd.DataFrame(scmdges['lval'])
    l=tl.drop_duplicates()
    l.columns=["f"]
    m = r.merge(l, how='left', indicator=True)
    e=m[m["_merge"]=="left_only"]
    exogenousVars=e['f'].values.tolist()
   
   #idenfitfy outcome varialbe = no outgoing arrows
    if interceptsCount>0:
       trS=scmdges[scmdges['intercept']==False]
       tr=pd.DataFrame(trS['lval'])
    else:   
       tr=pd.DataFrame(scmdges['lval'])
    r=tr.drop_duplicates()
    r.columns=["f"]
    tl=pd.DataFrame(scmdges['rval'])
    l=tl.drop_duplicates()
    l.columns=["f"]
    m = r.merge(l, how='left', indicator=True)
    e=m[m["_merge"]=="left_only"]
    outcomeVars=e['f'].values.tolist()
    
    #find vars not in data [i.e. latent]
    x=pd.DataFrame(scmdges[(scmdges['op'] == "~")])
    x=x[scmdges['intercept']==False]
    vars=pd.DataFrame(pd.unique(pd.concat([x['lval'],x['rval']],axis=0)))
    vars.columns=["var"]
    varsData=pd.DataFrame(dataVars)
    varsData.columns=["var"]
    j = vars.merge(varsData, on="var", how="left",indicator=True)
    latantVars = np.array(j[j['_merge'] == 'left_only']['var'])
    
    #vars in data and model
    vars = np.array(j[j['_merge'] == 'both']['var'])
    
    #define the DAG
    g = Digraph(name,format='png')
    g.attr(overlap='scale')
    g.attr(splines='True')

    g.attr('edge', fontsize='12')

    g.attr('node',fillcolor="white")
    g.attr('node',color='black')
    g.attr('node',shape='')
    g.attr('node',style="filled")
    g.attr('node',fontsize='12') 
   
    #latent variables
    g.attr('node',shape='circle')
    for lv in latantVars:
        g.node(lv,label=lv) 

    #exogenous variables
    g.attr('node',shape='triangle')
    for ev in exogenousVars:
        g.node(ev,label=ev) 

    #outcome variables
    g.attr('node',shape='invtriangle')
    for ov in outcomeVars:
        g.node(ov,label=ov)

    #nodes for variables in the data
    g.attr('node',shape='box')
    for dv in vars:
        g.node(dv,label=dv) 
 

    #Intercept variables
    if interceptsCount>0:
        g.attr('node',shape='plain')
        g.attr('node',fillcolor="white")
        g.attr('node',color="white")
        for iv in interceptVars:
            g.node(iv,label=iv)  
        
    #edges
    for index, row in scmdges.iterrows():
        if (np.isnan(row['p-value'])):
            pvalue = "*"
        else:
            pvalue = "p-value =" + "{:10.4f}".format(row['p-value'])+"\n Est. ="+ "{:10.2f}".format(row['Estimate'])  

        if (row['op'] == '~'):
            if (row['p-value'] < 0.05):
               g.edge(row['rval'], row['lval'], label=pvalue,  color="black", lblstyle="above, sloped")
            else:
               g.edge(row['rval'], row['lval'], label=pvalue, color=AvaOrangeLight, lblstyle="above, sloped")
        
    g.save()
    g.render(view=True)
    return(g)
    



def scm_induced_covariances_plot(dataVars, scmedges_in, name):
    ''' Good references
        https://graphviz.org/doc/info/shapes.html
        https://graphviz.readthedocs.io/en/stable/examples.html
        https://graphviz.org/doc/info/shapes.html
        https://stackoverflow.com/questions/66092429/how-to-format-edge-labels-in-graphviz

        Graph conventions:
        https://cjvanlissa.github.io/tidySEM/articles/sem_graph.html

        - Nodes
          - Exhogenous (no parents)             : Triangle
          - Variable (with parents and children): Square [box]
          - Latent variable (not in the data)   : Circle
          - Outcomes (no children)              : Inverted triangle
        - Edges
          - Causal/Regression efects: straingt arrow
          - Covariance: dashed curve
          - (Residual) variance: double headed arrow

        Param: dataVars - list of variables in the dataset
        Param: scmedges_in - Pandas dataframe similer to sempoy output
        Param: name - text to beused to name graph and resulting <name>.png  
        '''
    from graphviz import Digraph
    import copy
    import pandas as pd
    import numpy as np

    AvaOrangeLight = "#FF5800"
    AvaOrangeDark  = "#DC4600"
    AvaGoldLight = "#FFB414"
    AvaGoldDark  = "#E6A61C"

    scmdges = copy.copy(scmedges_in) # legacy from prev code
    scmdges=pd.DataFrame(scmdges[(scmdges['op'] == "~~")])
    
    scmdges1 = copy.copy(scmdges[scmdges['rval'] != scmdges['lval']])
    #idenfitfy exogenous varialbe = no incomming arrows
    tr=pd.DataFrame(scmdges1['rval'])
    r=tr.drop_duplicates()
    r.columns=["f"]
    tl=pd.DataFrame(scmdges1['lval'])
    l=tl.drop_duplicates()
    l.columns=["f"]
    m = r.merge(l, how='left', indicator=True)
    e=m[m["_merge"]=="left_only"]
    exogenousVars=e['f'].values.tolist()
   
   #idenfitfy outcome varialbe = no outgoing arrows
    tr=pd.DataFrame(scmdges1['lval'])
    r=tr.drop_duplicates()
    r.columns=["f"]
    tl=pd.DataFrame(scmdges1['rval'])
    l=tl.drop_duplicates()
    l.columns=["f"]
    m = r.merge(l, how='left', indicator=True)
    e=m[m["_merge"]=="left_only"]
    outcomeVars=e['f'].values.tolist()
    
    #find vars not in data [i.e. latent]
    x=pd.DataFrame(scmdges[(scmdges['op'] == "~~")])
    vars=pd.DataFrame(pd.unique(pd.concat([x['lval'],x['rval']],axis=0)))
    vars.columns=["var"]
    varsData=pd.DataFrame(dataVars)
    varsData.columns=["var"]
    j = vars.merge(varsData, on="var", how="left",indicator=True)
    latantVars = np.array(j[j['_merge'] == 'left_only']['var'])
    
    #vars in data and model
    vars = np.array(j[j['_merge'] == 'both']['var'])
    
    #define the DAG
    g = Digraph(name,format='png')
    g.attr(overlap='scale')
    g.attr(splines='True')

    g.attr('edge', fontsize='12')

    g.attr('node',fillcolor="white")
    g.attr('node',color='black')
    g.attr('node',shape='')
    g.attr('node',style="filled")
    g.attr('node',fontsize='12') 

    #latent variables
    g.attr('node',shape='circle')
    for lv in latantVars:
        #print("latent:",lv)
        g.node(lv,label=lv) 

    #exogenous variables
    g.attr('node',shape='triangle')
    for ev in exogenousVars:
        g.node(ev,label=ev) 

    #outcome variables
    g.attr('node',shape='invtriangle')
    for ov in outcomeVars:
        g.node(ov,label=ov)

    #nodes for variables in the data
    g.attr('node',shape='box')
    for dv in vars:
        g.node(dv,label=dv) 
        
    #edges
    for index, row in scmdges.iterrows():
        if (np.isnan(row['p-value'])):
            pvalue = "*"
        else:
            pvalue = "p-value =" + "{:10.4f}".format(row['p-value'])+"\n Est. ="+ "{:10.2f}".format(row['Estimate'])  

        if (row['op'] == '~~'):
                if (row['p-value'] < 0.05):
                   g.edge(row['rval'], row['lval'], label=pvalue,  dir="both", color="black", lblstyle="above, sloped")
                else:
                   g.edge(row['rval'], row['lval'], label=pvalue,  dir="both", color=AvaGoldLight, lblstyle="above, sloped")



    g.save()
    g.render(view=True)
    return(g)    