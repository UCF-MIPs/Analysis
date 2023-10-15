
def add_aggr_nets(df):
    '''
    From a dataframe of actors and influence-type edges, adds columns for aggregated influence 
    types (i.e. U->T from UM->TM, UM->TF, UF->TM, UF->TF)
    '''
    # Trust crossover
    #df['T_T'] = df['TF_TF'] + df['TF_TM'] + df['TM_TM'] + df['TM_TF']
    #df['U_U'] = df['UF_UF'] + df['UF_UM'] + df['UM_UM'] + df['UM_UF']
    #df['T_U'] = df['TF_UF'] + df['TF_UM'] + df['TM_UM'] + df['TM_UF']
    #df['U_T'] = df['UF_TF'] + df['UF_TM'] + df['UM_TM'] + df['UM_TF']
    
    # Same source
    df['TM_*'] = df['TM_UF'] + df['TM_UM'] + df['TM_TF'] + df['TM_TM']
    df['TF_*'] = df['TF_UF'] + df['TF_UM'] + df['TF_TF'] + df['TF_TM']
    df['UM_*'] = df['UM_UF'] + df['UM_UM'] + df['UM_TF'] + df['UM_TM']
    df['UF_*'] = df['UF_UF'] + df['UF_UM'] + df['UF_TF'] + df['UF_TM']

    # Same target
    df['*_TM'] = df['UF_TM'] + df['UM_TM'] + df['TF_TM'] + df['TM_TM']
    df['*_TF'] = df['UF_TF'] + df['UM_TF'] + df['TF_TF'] + df['TM_TF']
    df['*_UM'] = df['UF_UM'] + df['UM_UM'] + df['TF_UM'] + df['TM_UM']
    df['*_UF'] = df['UF_UF'] + df['UM_UF'] + df['TF_UF'] + df['TM_UF']

    return df
