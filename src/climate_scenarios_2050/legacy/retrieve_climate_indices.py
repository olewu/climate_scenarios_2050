# Download climate indices from climetedataguide

import argparse
from climate_scenarios_2050 import config
import requests
import pandas as pd
import numpy as np
from pathlib import Path

def extract_monthly(dl_extract_list:list) -> dict:
    """
    extract the data from a list of lists for each year containing the year and 12 values for each month
    """
    index_dict = {'date':[],'index_ts':[]}
    for line in dl_extract_list:
        for ii,element in enumerate(line):
            if ii != 0:
                try:
                    index_dict['index_ts'].append(float(element))
                    index_dict['date'].append(line[0] + '-' + str(ii).zfill(2))
                except:
                    print(element)
    return index_dict

def download_NAO_cdg(agg_code='djfm',itype='station',save_path=config.proj_base/'data/climate_indices',ret=False):
    """
    Download NAO index after Hurrell from Climate Data Guide (https://climatedataguide.ucar.edu)
    will automatically look for the latest update of the data
    different aggregations `agg_code` (`monthly`, `seasonal`, `annual`, `djf`, `djfm`, `mam`, `jja`, `son`) are available,
    but not for all index types `itype` (station-based `station` or pc-based `pc`)
    need to check the website for which ones are there
    will return a pandas dataframe with the time series when ret = True
    """
    
    update_date = pd.Timestamp('today').strftime('%Y-%m')

    # download the latest updated data:
    check = True
    while (check) and (pd.to_datetime(update_date) > pd.Timestamp('2015-01')): # don't look any further back than updated 10 years ago...
        response = requests.get(f'https://climatedataguide.ucar.edu/sites/default/files/{update_date}/nao_{itype}_{agg_code}.txt')
        check = response.status_code == 404
        if check:
            update_date = (pd.to_datetime(update_date) - pd.DateOffset(months=1)).strftime('%Y-%m')
    data = response.text
    extract = [[l for ii,l in enumerate(line.split())] for line in data.split('\n')[1:] if line]

    if agg_code == 'monthly':
        index_dict = extract_monthly(extract)
        years = pd.to_datetime(index_dict['date']).year
        start_NAO = years.min(); end_NAO =years.max()
    else:
        index_dict = {'year':[int(ex[0]) for ex in extract],'index_ts':[float(ex[-1]) for ex in extract]}
        start_NAO = min(index_dict['year']); end_NAO = max(index_dict['year'])

    # create dataframe and set missing values to nan:
    NAO = pd.DataFrame(index_dict)
    NAO = NAO.replace(-999.0,np.nan)
    
    # save to file:
    NAO.to_csv(save_path/f'NAO_{agg_code}_Hurrell_{itype}_based_{start_NAO}-{end_NAO}.csv',index=False)

    if ret:
        return NAO

def download_AMO_cdg(filtered=False,agg_code='annual',save_path=config.proj_base/'data/climate_indices',ret=False):
    """
    Download AMO index after Trenberth from Climate Data Guide (https://climatedataguide.ucar.edu)
    will automatically look for the latest update of the data
    two different aggregations `agg_code` (`monthly`, `annual`) are possible
    but the original data on the website is all monthly,
    possible to directly retrieve 10-year low-pass filtered data when filtered = True
    will return a pandas dataframe with the time series when ret = True
    """

    update_date = pd.Timestamp('today').strftime('%Y-%m')
    
    # download the latest updated data:
    check = True
    while (check) and (pd.to_datetime(update_date) > pd.Timestamp('2015-01')):
        if filtered:
            response = requests.get(f'https://climatedataguide.ucar.edu/sites/default/files/{update_date}/amo_monthly.10yrLP.txt')
            filt_ext = '_10yrLP-filter'
        else:
            response = requests.get(f'https://climatedataguide.ucar.edu/sites/default/files/{update_date}/amo_monthly.txt')
            filt_ext = ''
        check = response.status_code == 404
        if check:
            update_date = (pd.to_datetime(update_date) - pd.DateOffset(months=1)).strftime('%Y-%m')
    
    data = response.text
    extract = [[l for ii,l in enumerate(line.split())] for line in data.split('\n')[1:] if line]

    index_dict = extract_monthly(extract)

    AMO = pd.DataFrame(index_dict)
    AMO = AMO.replace(-999.0,np.nan)
    AMO['date'] = pd.to_datetime(AMO['date'])
    start_AMO = AMO.date.dt.year.min(); end_AMO = AMO.date.dt.year.max()
    if agg_code == 'annual':
        AMO = AMO.groupby(AMO['date'].dt.year).mean('year').reset_index().rename(columns={'date':'year'})
    elif agg_code != 'monthly':
        print('no option for anything other than monthly or annual aggregation implemented yet')
        return
    
    AMO.to_csv(save_path/f'AMO_{agg_code}{filt_ext}_HadISST1_Trenberth_{start_AMO}-{end_AMO}.csv',index=False)

    if ret:
        return AMO


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Download Climate Indexes from Climate Data Guide (https://climatedataguide.ucar.edu)")

    # Index argument:
    parser.add_argument('index', type=str, choices=['NAO','AMO'], help='Climate Index Name')

    # Save path argument:
    parser.add_argument('--save_path', type=str, help='Path to save climate index to')

    # Argument for temporal aggregation:
    parser.add_argument('--aggregation', nargs='?', default='djfm', type=str, help='Temporal Aggregation (currently only relevant for NAO), defaults to djfm', choices=['djfm','monthly','annual','seasonal','djf','mam','jja','son'])

    parser.add_argument('--index_type', nargs='?', default='station', type=str, choices=['station','pc']) # ,'PC_based'
    
    parser.add_argument('--filter', action='store_true', help='switch to download filtered data (only for AMO, 10-yr low-pass filter in that case)')

    args = parser.parse_args()

    index_name = args.index

    # check arguments
    if args.save_path is None:
        save_path = config.proj_base/f'data/climate_indices/{index_name}'
    else:
        save_path = Path(args.save_path)
    
    save_path.mkdir(exist_ok=True)

    if index_name == 'NAO':
        download_NAO_cdg(agg_code=args.aggregation,itype=args.index_type,save_path=save_path)
    elif index_name == 'AMO':
        download_AMO_cdg(filtered=args.filter,agg_code=args.aggregation,save_path=save_path)