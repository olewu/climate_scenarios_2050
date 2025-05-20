"""
Gets urls for cmip6 sst data from an esgf node given 
the experiment, data frequency, and variable. Removes
duplicate filenames that differ only in the grid label since
these are the same files just on diffferent grids. Default to 
native 'gn' grid. 
"""

from pyesgf.search import SearchConnection
# import xarray      as xr
import numpy       as np
from climate_scenarios_2050 import config
from collections   import defaultdict
import json
import re
from collections import defaultdict

# Input --------------------------------------------------------------
write2file     = True
project        = 'CMIP6' 
experiment_id  = 'ssp370'  #'ssp245' 'ssp370' 'historical'
frequency      = 'mon' 
variable       = 'psl' 
node           = 'dkrz'
time_range     = np.array((201501,209912)) #np.array((201501,209912)) np.array((201501,204912)) #np.array((185001,201412))
# --------------------------------------------------------------------

path_out  = {
    'historical': config.proj_base/'data/CMIP6/historical/url_repo/',
    'ssp245':  config.proj_base/'data/CMIP6/ssp245/url_repo/',
    'ssp370':  config.proj_base/'data/CMIP6/ssp370/url_repo/',
}

def get_data_urls_grouped_by_source(project, experiment_id, frequency, variable, node):
    """
    Queries the ESGF node for all available source_id (i.e., climate models)
    given the other constraints, then:
      - Prints how many unique source_ids are returned
      - Returns a dictionary with the structure:
            {
              <source_id1>: [ {'url': <download_url>, 'filename': <filename>}, ... ],
              <source_id2>: [ {...}, ... ],
              ...
            }
    """
    if node == 'ipsl':
        conn = SearchConnection('https://esgf-node.ipsl.upmc.fr/esg-search', distrib=True)
    elif node == 'dkrz':
        conn = SearchConnection('http://esgf-data.dkrz.de/esg-search', distrib=True)
    elif node == 'ceda':
        conn = SearchConnection('https://esgf.ceda.ac.uk/esg-search', distrib=True)
    
    facets = 'project,experiment_id,variable,frequency,latest,replica'

    query = conn.new_context(
        latest=True,
        replica=False,
        project=project,
        experiment_id=experiment_id,
        variable=variable,
        frequency=frequency,
        facets=facets
    )

    results_count = query.hit_count
    print(f"Search returned {results_count} dataset results")

    files_info = defaultdict(list)

    for idx, dataset in enumerate(query.search()):

        # 'source_id' could be a string or a list, depending on the dataset
        # We attempt to extract it safely.
        ds_source_raw = dataset.json.get('source_id', ['UNKNOWN_SOURCE'])

        # If it's a list, pick the first element; if it's already a string, use it directly.
        if isinstance(ds_source_raw, list) and len(ds_source_raw) > 0:
            ds_source_id = ds_source_raw[0]
        elif isinstance(ds_source_raw, str):
            ds_source_id = ds_source_raw
        else:
            ds_source_id = "UNKNOWN_SOURCE"

        # Now ds_source_id is a string, safe to use as a dict key
        files_list = dataset.file_context().search()
        for f in files_list:
            files_info[ds_source_id].append({
                'url': f.download_url,
                'filename': f.filename
            })

        print(f"Processed dataset {idx+1}/{results_count} - source_id={ds_source_id}")

    # Summarize unique source IDs
    unique_source_ids = list(files_info.keys())
    print(f"\nNumber of unique source_ids found: {len(unique_source_ids)}")
    print(f"These source_ids are: {unique_source_ids}")

    return files_info


def filter_grid_labels(file_dicts):
    """
    Given a list of dicts (each with 'filename' and 'url'), remove duplicates that
    differ only by the grid label (_gn, _gr, _gr1...). Specifically:
      - If a group contains any 'gn' (native grid), keep only 'gn' files.
      - Otherwise keep the entire group.

    Prints which files are kept vs. discarded if duplicates exist.
    """
    import re
    from collections import defaultdict
    
    groups = defaultdict(list)
    # This pattern will match _gn_, _gr_, _gr1_, ...
    grid_label_pattern = re.compile(r'_(gn|gr\d*)_')
    
    for fdict in file_dicts:
        filename = fdict['filename']
        match = grid_label_pattern.search(filename)
        
        if match:
            # Example label: 'gn', 'gr', 'gr1', etc.
            label = match.group(1)
            matched_segment = match.group(0)   # e.g. '_gn_', '_gr1_', ...
            base_name = filename.replace(matched_segment, '_<<<GRID>>>_')
        else:
            label = None
            base_name = filename
        
        groups[base_name].append((label, fdict))
    
    filtered = []
    for base_name, items in groups.items():
        labels = [lbl for (lbl, _) in items]
        
        # If there's more than 1 file in a group, it's a "duplicate" group
        if len(items) > 1:
            print(f"Found duplicates for base name: '{base_name}'")
            print("All files:")
            for lbl, fdict in items:
                print(f"  - label={lbl}  filename={fdict['filename']}")
        
        if any(lbl == 'gn' for lbl in labels):
            # Keep only the 'gn' ones
            kept      = [f for (lbl, f) in items if lbl == 'gn']
            discarded = [f for (lbl, f) in items if lbl != 'gn']
        else:
            # If no gn, keep them all
            kept      = [f for (_, f) in items]
            discarded = []
        
        # Print the keep/discard if duplicates exist
        if len(items) > 1:
            kept_files      = [f["filename"] for f in kept]
            discarded_files = [f["filename"] for f in discarded]
            print(f"  -> Keeping   : {kept_files}")
            print(f"  -> Discarding: {discarded_files}")
            print("------")
        
        filtered.extend(kept)
    
    return filtered



def filter_time_range_partial_overlap(file_dicts, min_start=185001, max_end=201412):
    """
    Given a list of dicts (each with 'filename' and 'url'), keep only the files
    whose date range (YYYYMM-YYYYMM) intersects the interval [185001, 201412].
    
    - Looks for the pattern '_YYYYMM-YYYYMM.nc' in the filename.
    - If found, parses the start/end as integers.
    - We check for overlap: a file covers [start_int, end_int].
      The desired interval is [min_start, max_end].
      They overlap if (start_int <= max_end) and (end_int >= min_start).
    - If they do NOT overlap, we discard the file.
    - If the pattern is not found, we discard the file (you can change this behavior).
    
    Returns a new filtered list of file dicts.
    Prints which files are discarded, for clarity.
    """
    pattern = re.compile(r'_(\d{6})-(\d{6})\.nc$')

    kept      = []
    discarded = []

    for fdict in file_dicts:
        fname = fdict["filename"]
        match = pattern.search(fname)
        if match:
            start_str, end_str = match.groups()  # e.g. '185001', '201412'
            start_int = int(start_str)
            end_int   = int(end_str)

            # Check if [start_int, end_int] intersects [min_start, max_end].
            # Intersection occurs if:
            #   start_int <= max_end AND end_int >= min_start
            if start_int > max_end or end_int < min_start:
                # no overlap => discard
                discarded.append(fdict)
            else:
                kept.append(fdict)
        else:
            # If the pattern is not found, decide whether to keep or discard.
            # Here, we discard if we can't parse the time range.
            discarded.append(fdict)

    if discarded:
        print(f"Discarding {len(discarded)} files that do NOT overlap {min_start}-{max_end} (or missing time pattern):")
        for d in discarded:
            print(f"  - {d['filename']}")

    return kept



def filter_latest_version(file_dicts):
    """
    Given a list of dicts, each with a 'url' that includes something like /vYYYYMMDD/,
    keep only the file(s) with the highest version for each unique filename base.
    """
    
    # Pattern that matches 'v20200724' or 'v20190914' etc.
    version_pattern = re.compile(r'/v(\d{8})/')
    grouped         = defaultdict(list)
    
    for fdict in file_dicts:
        filename = fdict['filename']
        url = fdict['url']
        match = version_pattern.search(url)
        if match:
            version_str = match.group(1)  # e.g. '20200724'
            version_int = int(version_str)
        else:
            version_int = -1  # or 0, if there's no version folder, treat as minimal?
        
        grouped[filename].append((version_int, fdict))
    
    filtered = []
    for fn, versioned_files in grouped.items():
        # Sort by version descending
        versioned_files.sort(key=lambda x: x[0], reverse=True)
        # Keep only the top version
        highest_version, best_dict = versioned_files[0]
        filtered.append(best_dict)
    
    return filtered


def filter_files(files_info,time_range):
    """
    For each source_id's file list, 
    1) filter out files completely outside my desired time range.
    2) filter out files that are the same but have different versions
    3) filter out files with different gridtype labels,
    preferring 'gn' (native curvlinear) over 'gr'/'gr1' (rectilinear) etc.

    Modifies files_info in-place and returns it.
    """
    for source_id, file_dicts in files_info.items():
        print(f"\nProcessing source_id = {source_id} ...")

        # Step 1: Filter by time range
        filtered_files_info = filter_time_range_partial_overlap(file_dicts, time_range[0], time_range[-1])

        # step 2: filter latest version of simulations
        filtered_files_info = filter_latest_version(filtered_files_info)

        # Step 3: Filter by grid labels
        filtered_files_info = filter_grid_labels(filtered_files_info)
        
        # Assign back to the dictionary
        files_info[source_id] = filtered_files_info
    
    return files_info


def write_out_urls_to_json(path_out,experiment_id,variable,project,frequency,files_info,node,write2file):
    path_out_local = path_out[experiment_id] / variable
    filename_out   = 'file_urls_esgf_node_' + node + '_' + project + '_' + experiment_id + '_' + frequency + '_' + variable + '.json'
    with open(path_out_local / filename_out, 'w') as f:
        json.dump(files_info, f, indent=2)
    return


# main code
if __name__ == "__main__":

    # get urls
    files_info  = get_data_urls_grouped_by_source(project, experiment_id, frequency, variable, node)
    
    # filter out files that are outside time range, same file different versions,
    # and same file different grids, preferring native curvelinear grid instead of rectilinear
    files_info = filter_files(files_info,time_range)
    
    # Write out urls to JSON file for later use
    write_out_urls_to_json(path_out,experiment_id,variable,project,frequency,files_info,node,write2file)