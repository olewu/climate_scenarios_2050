import pandas as pd
import subprocess as sbp

from climate_scenarios_2050.config import proj_base

import requests
import re
import json
from time import sleep
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# node: "https://esgf-metagrid.cloud.dkrz.de/search" does not work like this

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    # files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)

def get_simulation(experiment,variable,model,member,activity=None):
    
    if activity is None:
        if experiment == 'historical':
            activity = 'CMIP'
        elif 'ssp' in experiment:
            activity = 'ScenarioMIP'
        else:
            activity = ''
    
    fail = True; tries = 0
    while fail and (tries < 15):
        tries += 1
        try:
            result = esgf_search(
                    files_type='HTTPServer',
                    table_id='Amon',
                    variable_id=variable,
                    activity_id=activity,
                    source_id=model,
                    experiment_id=experiment,
                    variant_label=member,
                    latest=True # get only latest version
            )
            fail=False
        except Exception as e:
            sleep(10)
            fail=True

    if fail:
        print(f'search for data keeps failing with {e}')
        return {},''

    if not result:
        print(f'cannot find data for {experiment} {variable} {model} {member}; returning empty')
        return {},''

    # check result for duplicates (from different nodes):
    file_name = list(set([Path(res).name for res in result]))
    node_name = list(set([Path(res).parent for res in result]))

    path_out = proj_base/f'data/CMIP6/{experiment}/{model}/{variable}'
    path_out.mkdir(exist_ok=True,parents=True)

    dl_success = {fi:'failed' for fi in file_name}
    for node in node_name:
        for fi in file_name:
            if dl_success[fi] == 'failed':
                dl_file = str(node/fi).replace('http:/','http://').replace('https:/','https://')
                wget = sbp.run(
                    [
                        'wget','-c',
                        '-T', '120', # retry after 120 s
                        '-t', '3', # retry 3 times
                        '-P', path_out,
                        str(dl_file),
                    ]
                )

                if wget.returncode == 0:
                    dl_success[fi] = 'success'
    return dl_success, path_out

def update_status(dl_success,model,member,path_out,experiment,new_status):
    if len(dl_success) > 1:
        mergefiles = [path_out/key for key,val in  dl_success.items() if val == 'success']
        start_date = min([datetime.strptime(re.search('_(\d{6})-(\d{6})$',mefi.stem).group(1),dt_fmt) for mefi in mergefiles]).strftime(dt_fmt)
        end_date = max([datetime.strptime(re.search('_(\d{6})-(\d{6})$',mefi.stem).group(2),dt_fmt) for mefi in mergefiles]).strftime(dt_fmt)
        replc = re.search('(\d{6}-\d{6})$',mergefiles[0].stem).group(1)
        outfile = str(path_out/mergefiles[0].name.replace(replc,f'{start_date}-{end_date}'))
        mrg = sbp.run([
            'ncrcat',
            '-O',
            '-h',
            *[str(mefi) for mefi in mergefiles],
            outfile,
        ])
        if mrg.returncode == 0:
            [mefi.unlink() for mefi in mergefiles]
            status = 'done'
    elif [val for _,val in dl_success.items()] == ['success']:
        status = 'done'
    else:
        status = 'failed'

    new_status.loc[(new_status.model==model) & (new_status.member_id == member),f'{variable}_{experiment}'] = status

    return new_status.copy(),status

def ensure_variable_in_path(filepath):
    filepath = Path(filepath)  # Convert string to Path object
    filename = filepath.name  # Extract filename
    parent_dir = filepath.parent  # Extract parent directory

    # Regex to extract variable name (before "_Amon_")
    match = re.match(r"([^_]+)_[A,O]mon_", filename)
    if not match:
        return str(filepath)  # No variable name found, return original path

    variable_name = match.group(1)  # Extracted variable name

    # Check if the parent directory already contains the variable name
    if parent_dir.name == variable_name:
        return str(filepath)  # No modification needed

    # Construct new path by inserting the variable directory
    new_path = parent_dir / variable_name / filename
    
    return str(new_path)


dt_fmt = '%Y%m'

#---------CLEAN UP WHAT'S ALREADY THERE---------#
# iterate through the current archive and find simulations consisting of multiple
# files. Merge those.

cmip_path = Path('/datalake/NS9873K/DATA/CMIP6/')

# find all netcdf files:
all_nc = sorted(list(cmip_path.rglob('*.nc')))

# sort them into groups (unique model and member_id):
model_member_id = defaultdict(list)
for nc in all_nc:

    # check if the file is empty, delete in that case and skip in the sorting:
    if nc.stat().st_size == 0:
        nc.unlink()
        continue

    exp = re.search('CMIP6/([^/]+)/',str(nc.parent)).group(1)
    var,model,member_id = re.search('([^_]+)_[A,O]mon_([^_]+)_.*?_(r\d+i\d+p\d+f\d+)_',nc.stem).groups()
    model_member_id[f'{exp}_{var}_{model}_{member_id}'].append(nc)

for mod_mem, files in model_member_id.items():
    # if multiple files exist, merge them:
    if len(files) > 1:
        start_date = min([datetime.strptime(re.search('_(\d{6})-(\d{6})$',fi.stem).group(1),dt_fmt) for fi in files]).strftime(dt_fmt)
        end_date = max([datetime.strptime(re.search('_(\d{6})-(\d{6})$',fi.stem).group(2),dt_fmt) for fi in files]).strftime(dt_fmt)
        replc = re.search('(\d{6}-\d{6})$',files[0].stem).group(1)
        outfile = files[0].parent/files[0].name.replace(replc,f'{start_date}-{end_date}')
        # make sure output is saved in the variable directory:
        outfile = ensure_variable_in_path(outfile)
        X = len(files)
        print(f'merging {X} files for {mod_mem} into {outfile}')
        mrg = sbp.run([
            'ncrcat',
            '-O',
            '-h',
            *[str(fi) for fi in files],
            outfile
        ])
        if mrg.returncode == 0:
            [fi.unlink() for fi in files]


#---------GET MORE SIMULATIONS BASED ON MARIKO'S LIST---------#
status_df = pd.read_csv(proj_base/'data/CMIP6_download_for_Climate_Futures.csv')

# ssp370_cols = [col.replace('historical','ssp370') for col in list(status_df) if 'historical' in col]
# make a new set of columns for ssp370 (all empty):
# status_df[ssp370_cols] = np.nan

# new status file with updates recorded:
new_status = status_df.copy()

variable = 'pr'
exclude_members = [('GISS-E2-1-G','r3i1p3f1'),]
# go line by line through the status file to complete the updates:
try:
    for _,line in status_df.iterrows():
        model = line.model
        member = line.member_id
        # if model == 'GISS-E2-1-G':
        #     continue
        print(model,member)
        # check status of psl historical:
        if line[f'{variable}_ssp370'] not in ['done','copied','copeid']:
            dl_success, path_out = get_simulation(experiment='ssp370',variable=variable,model=model,member=member)
            # if multiple files were downloaded, merge them
            new_status,stat_ssp370 = update_status(dl_success,model,member,path_out,'ssp370',new_status)
        if (line[f'{variable}_ssp370'] in ['done','copied','copeid']) or (stat_ssp370 == 'done'):
            if line[f'{variable}_historical'] not in ['done','copied','copeid']:
                dl_success, path_out = get_simulation(experiment='historical',variable=variable,model=model,member=member)
                # if multiple files were downloaded, merge them
                new_status,stat_hist = update_status(dl_success,model,member,path_out,'historical',new_status)
            
            if line[f'{variable}_ssp245'] not in ['done','copied','copeid']:
                # if model == 'GISS-E2-1-G':
                # if (model,member) in exclude_members:
                #     continue
                dl_success, path_out = get_simulation(experiment='ssp245',variable=variable,model=model,member=member)
                # if multiple files were downloaded, merge them
                new_status,stat_ssp245 = update_status(dl_success,model,member,path_out,'ssp245',new_status)
            
except Exception as exc:
    print(exc)
    pass

new_status.to_csv(proj_base/'data/CMIP6_download_for_Climate_Futures.csv',index=False)


#--------find all available ssp370 simulations---------#

# find all available ssp370 simulations:
pathway = 'ssp245'
result = esgf_search(
        files_type='HTTPServer',
        table_id='Amon',
        variable_id=variable,
        activity_id='ScenarioMIP',
        experiment_id=pathway,
        latest=True # get only latest version
)

model_member_id = defaultdict(list)
for res in result:
    var,varty,model,exp,member_id,grid,span = Path(res).stem.split('_')
    model_member_id[f'{exp}_{var}_{model}_{member_id}'].append(res)

# save the output list for later reference
with open(proj_base/f"data/all_available_{pathway}.json", "w") as fp:
    json.dump(model_member_id , fp)

# based on the available ssp370 experiments, download ssp370 and corresponding historical simulations
# first compare with what is already there, i.e. contents of "data/CMIP6_download_for_Climate_Futures.csv"
rest_cols = {col:[np.nan] for col in list(status_df) if col not in ['model','member_id']}
for mkey,files in model_member_id.items():
    model,member = mkey.split('_')[-2:]
    if status_df.loc[(status_df.model==model)&(status_df.member_id==member)].empty:
        new_row = pd.DataFrame({"model": [model], "member_id": [member]} | rest_cols)
        new_status = pd.concat([new_status,new_row],ignore_index=True)

new_status = new_status.sort_values(['model','member_id'])
new_status.to_csv(proj_base/'data/CMIP6_download_for_Climate_Futures_update.csv',index=False)
