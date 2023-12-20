

def get_mt_computation_dict():
    '''
    elementID < indicing style
    laymans < MT component in seismic terminology
    pq < indices of the two component parameters
    ODscaler < off-diagonal scaler, allows multiplication of OD elements by 2 so can work on 6 MT components as opposed to 9
    MCweighting < for weighting MC component importance [TO-DO]

    '''
    # Pre-make dictionary of the MT components values
    MT_comp_dict = [{'elementID': 0, 'laymans': 'xx', 'pq': [0, 0], 'ODscaler': 1, 'MCweighting': 1},
                    {'elementID': 1, 'laymans': 'yy', 'pq': [1, 1], 'ODscaler': 1, 'MCweighting': 1},
                    {'elementID': 2, 'laymans': 'zz', 'pq': [2, 2], 'ODscaler': 1, 'MCweighting': 1},
                    {'elementID': 3, 'laymans': 'xy', 'pq': [0, 1], 'ODscaler': 2, 'MCweighting': 1},
                    {'elementID': 4, 'laymans': 'xz', 'pq': [0, 2], 'ODscaler': 2, 'MCweighting': 1},
                    {'elementID': 5, 'laymans': 'yz', 'pq': [1, 2], 'ODscaler': 2, 'MCweighting': 1},
                    ]

    return MT_comp_dict

