"""Curates specimen metadata from 'Wing Scans Specimen List.xls' in order to make
    identification targets. Saves data to `metadata_path`."""

### Setup
print ('\nSetting up ...')

import os, sys, re
import pandas as pd
import numpy as np

def str2num(inpt,output_cast=int):
    """Extract the first number in a string containing one or more multi-digit
    numbers.

    Parameters
    ----------
    inpt : str
        Input string. Can be unicode or not.
    output_cast : {int,float,str,etc.}
        How the output should be cast. Default is int.

    Returns
    -------
    output : int (or other depending on value of *output_cast*)
        A number representing the first single- or multi-digit number
        encountered in *string*, cast according to value of *output_cast*.
        Raises error if *inpt* is not of type str.
    """
    if type(inpt) not in (str,unicode,np.string_):
        raise RuntimeError('Expecting a string.')
    s = unicode(inpt).strip()
    r = re.findall(r'\d+',s) #Find all groups of 1+ digits
    if len(r) == 0: return None #If none, return None
    else: return output_cast(r[0]) #Otherwise return the first group of digits,
    #cast as *output_cast*

# Load config file, if not already done
try: config_loaded
except NameError:
    f = 'config_linux.txt' if 'linux' in sys.platform else 'config.txt'
    execfile(os.path.join(os.getcwd(),f))


### Get data
# Import original metadata from spreadsheet
print ('Fetching original metadata ...')
orig_metadata_path = 'D:\Dropbox\Rutgers\Research\Wing scans\Wing Scans Specimen List.xlsx'
md = pd.read_excel(orig_metadata_path,sheetname=0,header=0,index_col=None)

# Get list of image filenames
print ('Fetching image filenames ...')
image_files = [f for f in os.listdir(image_path) if f.endswith('.tif')]


### Clean data
print ('Cleaning ...')

# Curate taxonomic names
md['Genus'] = md['Genus'].str.title()
md['Species'] = md['Species'].str.lower()
md['Sex'] = md['Sex'].str.upper()
md['Suborder'] = md['Suborder'].str.title()

# Quality control specimens
md = md[md['Suborder'].isin(['Anisoptera','Zygoptera'])] #Filter out non-odes
md = md[md['Sex'].isin(['F','M'])] #Filter unsexed
# Filter un-identified
md = md[~(md['Species'].str.contains('cf|\.|\?|spp|unk',na=True)|(md['Species'].isin(['sp',' '])))]
# Remove subspecies, keep only specific epithet
md.Species = md['Species'].apply(lambda x: x.split(' ')[0])
# Filter 'sp1','sp2',etc
md = md[~md['Species'].str.match(r'sp[\d]',na=False)]
# Make 'numCode' col: #### from WRK-WS-#### in 'Scan ID' col:
md['numCode'] = map(str2num, md['Scan ID'].values)

# Add an img_filename column by matching image filenames to Scan IDs
imgNumCode2filename = dict(zip( map(str2num,image_files), image_files))
def get_fn(numCode): #Catch numCodes in metadata file but not image filenames
    try: return imgNumCode2filename[numCode]
    except KeyError: return None
md['img_filename'] = map(get_fn,md['numCode'].values)
# Drop specimens that lack a filename
md.dropna(axis=0,subset=['img_filename'],inplace=True)

# Combine several columns of 'codes' into a single column called `Codes`
codes = md[['RU Collection #','Code','Other code','Locality code']]
def combine_codes(row):
    row = row.dropna()
    if len(row)==0: return ''
    else: return '; '.join(row)
codes = codes.apply(combine_codes,axis=1)
md['Codes'] = codes

# Export clean metadata
print ('Saving ...')
columns_to_save = ['img_filename','Scan ID','Suborder','Family','Genus','Species','Sex',
                   'Collection Date','Country','State/Province','County',
                   'Collection Locality','GPS Coordinates','Collector',
                   'Codes','Condition']
md.to_csv(metadata_path,columns=columns_to_save,index=False,header=True,
          encoding='utf-8')
print ('Done. Clean metadata save to: {}'.format(metadata_path))
