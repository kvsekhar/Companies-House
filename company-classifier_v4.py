# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:56:57 2020

@author: RaviTejaPekala
"""

# Import Libraries
import sys
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
pd.options.mode.chained_assignment = None

START_TIME = time()

FILE = sys.argv[1].split('.')[0]+'-cleaned.csv'
OUT_FILE = FILE.split('.')[0]
print(f"Using {FILE} generated from previous script")

# Import Data
#import DATA
DATA = pd.read_csv(FILE)

# Don't need first 3 columns
try:
    DATA.drop(
        ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'],
        axis=1, inplace=True)  # drop columns by index
except KeyError as e:
    print("Error:", e)    
# Data Cleaning
    # Repair Company Number
DATA['Company Registered Number'] = DATA[
    'Company Registered Number'].apply(lambda x: str(x).zfill(8))

# Strip everything else and save only Year
DATA['Year'] = DATA['Year'].apply(lambda x: str(x).strip())

# Clean the Company Name
#DATA['Company Name'] = DATA['Company Name'].apply(lambda x: str(x).strip())

# Total Number of rows in the dataset
print(f'Total number of rows in the File: {DATA.shape[0]}')
DATA = DATA.drop_duplicates()   # remove duplicate entries
print(f'Total number of rows after removing the duplicates: {DATA.shape[0]}')

# Groupby Company Numbers
company_groups = DATA.groupby(['Company Registered Number'])
unique_companies = company_groups.groups
print(f"Total Number of rows with Unique company Numbers: {len(unique_companies)}")

# Employee Data
grouped_companies_employees = pd.DataFrame({
    'count' : DATA.groupby([
        'Company Registered Number', 'Number of Employees']).size()}).reset_index()
emp_num = grouped_companies_employees.groupby(['Company Registered Number']).groups
print(f"Companies with Employee Data: {len(emp_num)}")

# Equity Data
grouped_companies_equity = pd.DataFrame({
    'count' : DATA.groupby([
        'Company Registered Number', 'Equity']).size()}).reset_index()
equi_num = grouped_companies_equity.groupby(['Company Registered Number']).groups
print(f"Companies with Equity Data: {len(equi_num)}")

# Shareholder Data
grouped_companies_shareholder = pd.DataFrame({
    'count' : DATA.groupby([
        'Company Registered Number', 'Shareholder Funds']).size()}).reset_index()
share_num = grouped_companies_shareholder.groupby(['Company Registered Number']).groups
print(f"Companies with Shareholder Data: {len(share_num)}")

# After Replacing NaN Equity Data with Shareholder Funds Data
null_equity_before = DATA[[
    'Equity', 'Shareholder Funds', 'Company Registered Number']].groupby([
        'Company Registered Number']).filter(lambda x: x['Equity'].isnull().all())
grouped_companies_equity_before = pd.DataFrame({
    'count' : DATA.groupby([
        'Company Registered Number', 'Equity']).size()}).reset_index()

null_equity_after = null_equity_before.copy()
null_equity_after['Equity'] = null_equity_after.apply(
    lambda row: row['Shareholder Funds'] if np.isnan(
        row['Equity']) else row['Equity'], axis=1)

grouped_companies_equity_after = pd.DataFrame({
    'count' : null_equity_after.groupby([
        'Company Registered Number', 'Equity']).size()}).reset_index()

before_eq = grouped_companies_equity_before.groupby(
    ['Company Registered Number']).groups
after_eq = grouped_companies_equity_after.groupby(
    ['Company Registered Number']).groups
print(f"Companies with Equity Data before replacing: {len(before_eq)}")
print(f"Companies with Equity Data after replacing: {len(before_eq) + len(after_eq)}")

# Turnover Revenue
grouped_companies_revenue = pd.DataFrame({
    'count' : DATA.groupby([
        'Company Registered Number', 'Turnover Revenue']).size()}).reset_index()
revenue_num = grouped_companies_revenue.groupby(['Company Registered Number']).groups
print(f"Companies with Turnover Revenue Data: {len(revenue_num)}")

# SME Classification
# Select relevant attribute and fill equity with shareholder funds
sme = DATA[
    ['Company Registered Number',
     'Equity', 'Shareholder Funds',
     'Year', 'Turnover Revenue', 'Number of Employees']]
sme['Equity'].fillna(sme['Shareholder Funds'], inplace=True)
sme_original = sme.copy()
tqdm.pandas(desc="Progress")

# Number of companies with Equity Data
sme_equity = sme[sme['Equity'].notnull()]
print('Number of companies with Equity Data',
      len(set(sme_equity['Company Registered Number'])))

# Number of companies with NO Equity Data
l = list(set(sme_equity['Company Registered Number']))
sme_noequity = sme[~sme['Company Registered Number'].isin(l)]
print('No. of companies with No Equity Data',
      len(set(sme_noequity['Company Registered Number'])))
# List of SME Group 1, if previous check is validated
list_sme1 = list(set(sme_noequity['Company Registered Number']))

# Check other attributes for companies with No Equity Data
#print(sme_noequity[sme_noequity['Net assets (liabilities)'].notnull()].max())
print(sme_noequity[sme_noequity['Turnover Revenue'].notnull()].max())
print("-#"*40)
print(
    sme_noequity[sme_noequity['Turnover Revenue'].notnull() |
                 sme_noequity['Number of Employees'].notnull()].head())

# Number of companies with All 3 Attributes: Equity, Turnover, Employees
sme.sort_values(by=['Number of Employees', 'Turnover Revenue'], ascending=False)
sme_all3 = sme[sme['Equity'].notnull() &
               sme['Number of Employees'].notnull() &
               sme['Turnover Revenue'].notnull()]
sme_all3 = sme_all3.sort_values(
    by=['Number of Employees', 'Turnover Revenue'], ascending=False)
sme2 = sme_all3
print(
    'Companies with Equity, Turnover & Employee Data: ',
    len(set(sme_all3['Company Registered Number'])))

#List of Non-SME companies, if any two conditions is met
sme_large = sme2[
    ((sme2['Turnover Revenue'] >= 36000000.0) & (sme2['Equity'] >= 18000000.0)) | \
    ((sme2['Number of Employees'] >= 250)& (sme2['Turnover Revenue'] >= 36000000.0)) |  \
    ((sme2['Equity'] >= 18000000.0) & (sme2['Number of Employees'] >= 250))]
print('Number of Non-SME Group 1: ', len(set(sme_large['Company Registered Number'])))

list_large1 = list(set(sme_large['Company Registered Number']))

# List of SME Group 2, not satisfying any two conditions
list_sme2 = list(
    set(sme_all3['Company Registered Number'])^set(sme_large['Company Registered Number']))

# Check for intersection,
list_all3 = list(set(sme_all3['Company Registered Number']))
list_noequity = list(set(sme_noequity['Company Registered Number']))
list_common = list(set(list_all3).intersection(set(list_noequity)))
#sme_noequity[sme_noequity['Company Registered Number'].isin(list_common)]
#sme_all3[sme_all3['Company Registered Number'].isin(list_common)]
list_remaining = list(set(list_all3 + list_noequity))
list_14k = list(set(sme['Company Registered Number']) ^ set(list_all3 + list_noequity))
print('Number of Remaining Companies: ', len(list_14k))

# Check for Remaining companies attributes
sme_14k = sme[sme['Company Registered Number'].isin(list_14k)]
print('Maximum Turnover in Remaining Companies: ', sme_14k['Turnover Revenue'].max())
print('Maximum Equity in Remaining Companies: ', sme_14k['Equity'].max())
print('Maximum Employees in Remaining Companies: ', sme_14k['Number of Employees'].max())

# Equity and Turnover companies
sme_14 = sme_14k.sort_values(by=['Equity', 'Turnover Revenue'], ascending=False)
sme_equity_turnover = sme_14[sme_14['Equity'].notnull() &
                             sme_14['Turnover Revenue'].notnull()]
#sme_equity_turnover = sme[sme['Equity'].notnull() & sme['Number of Employees'] !=0]
print('Companies with Equity and Turnover: ',
      len(set(sme_equity_turnover['Company Registered Number'])))

# List of Non-sme companies - group 2
ll = sme_equity_turnover[
    (sme_equity_turnover['Equity'] > 18000000.0) &
    (sme_equity_turnover['Turnover Revenue'] > 36000000.0)]
list_large2 = list(set(ll['Company Registered Number']))

# List of SME Group 3
list_sme3 = list(
    set(sme_equity_turnover['Company Registered Number'])^
    (set(ll['Company Registered Number'])))

# Equity and Employees
sme_14 = sme_14k.sort_values(by=['Equity', 'Number of Employees'], ascending=False)
sme_equity_employees = sme_14[sme_14['Equity'].notnull() &
                              sme_14['Number of Employees'].notnull()]
print('Companies with Equity and Employees: ',
      len(set(sme_equity_employees['Company Registered Number'])))

# Non-SME Group 3
lll = sme_equity_employees[(sme_equity_employees['Equity'] > 18000000.0) &
                           (sme_equity_employees['Number of Employees'] > 250)]
list_large3 = list(set(lll['Company Registered Number']))
#SME Group 4
list_sme4 = list(set(sme_equity_employees['Company Registered Number'])^
                 (set(lll['Company Registered Number'])))

#Turnover and Employees
sme_14 = sme_14k.sort_values(by=['Turnover Revenue', 'Number of Employees'], ascending=False)
sme_turnover_employees = sme_14[sme_14['Turnover Revenue'].notnull() &
                                sme_14['Number of Employees'].notnull()]
print('Companies with Turnover and Employee Data: ',
      len(set(sme_turnover_employees['Company Registered Number'])))

# Less than two attributes available
print('-#'*40)
print('Companies with any one variable available: \n')
list_et = list(sme_equity_turnover['Company Registered Number'])
list_ee = list(sme_equity_employees['Company Registered Number'])
list_ete = list(set(list_et+list_ee))
print('Number of Companies: ', len(
    set(sme_14k['Company Registered Number']) ^ set(list_ete)))

list_9k = list(set(sme_14k['Company Registered Number']) ^ set(list_ete))
sme_9k = sme_14k[sme_14k['Company Registered Number'].isin(list_9k)]
print('Maximum Turnover: ', sme_9k['Turnover Revenue'].max())
print('Maximum Equity: ', sme_9k['Equity'].max())
print('Maximum Shareholder Fund: ', sme_9k['Shareholder Funds'].max())
print('Maximum Number of Employees: ', sme_9k['Number of Employees'].max())
#print(sme_9k['Net assets (liabilities)'].max())

# List SME Group 5
print('-#'*20)
print("Companies that are not classified yet:")
list_notconfirmed = sme_9k[
    (sme_9k['Equity'] > 18000000.0) |\
    (sme_9k['Number of Employees'] > 250) |\
    (sme_9k['Turnover Revenue'] > 36000000.0) |\
    (sme_9k['Shareholder Funds'] > 18000000.0)]
list_notconfirmed = list(set(list_notconfirmed['Company Registered Number']))
print("Number of Companies: ", len(list_notconfirmed))

# List SME Group 5
list_sme5 = list(set(sme_9k['Company Registered Number'])^set(list_notconfirmed))

# Total Number of SME
list_sme = list(set(list_sme1+list_sme2 + list_sme3 + list_sme4))
print('Total Number of SME Companies: ', len(list_sme))

# Total Number of Non-SME
list_notsme = list_large1 +list_large2 +list_large3
print('Total Number of Non-SME Companies: ', len(set(list_notsme)))

# Final Classification
print('Final SME Classification..')
list_unknown = list_sme5
sme = pd.DataFrame()
notsme = pd.DataFrame()
unknown = pd.DataFrame()
check = pd.DataFrame()
if len(list_sme) > 0:
    sme = pd.DataFrame(list_sme)
    sme.columns = ['Company Number']
    sme['Type'] = 'SME'
if len(set(list_notsme)) > 0:
    notsme = pd.DataFrame(list_notsme)
    notsme.columns = ['Company Number']
    notsme['Type'] = 'Not SME'
if len(set(list_unknown)) > 0:
    unknown = pd.DataFrame(list_unknown)
    unknown.columns = ['Company Number']
    unknown['Type'] = 'Probably SME'
if len(list_notconfirmed) > 0:
    check = pd.DataFrame(list_notconfirmed)
    check.columns = ['Company Number']
    check['Type'] = 'Probably not SME'

# Company with SME-Type Dataframe
Final = sme.append(notsme).append(unknown).append(check)
sme_original_de_duplicated = sme_original.drop_duplicates(
    subset='Company Registered Number')
sme_type = sme_original_de_duplicated[
    'Company Registered Number'].apply(
        lambda x: Final[Final['Company Number'] == x].iloc[0] if x in list(
            Final['Company Number']) else pd.Series())

Final_df = pd.concat([sme_original_de_duplicated, sme_type], axis=1)
Final_df = Final_df.reset_index(drop=True)
Final_df.drop(Final_df.columns[[0]], inplace=True, axis=1)

# Rename Columns
Final_df.rename(columns={
    'Company Number': 'Company Registered Number', 'Type': 'SME Classification'},
                inplace=True)

print("SME Classification done!")

# Classification by Equity
sorted_data = DATA.sort_values(by=['Company Registered Number', 'Year'])

# Fill in NaN Equity values with available Shareholder Funds
sorted_data['Equity'].fillna(sorted_data['Shareholder Funds'], inplace=True)
print("Replacing Null Equity with Shareholder Funds")

# Drop NaN Equities
sorted_data = sorted_data.dropna(subset=['Equity']).reset_index()

# Clean the data for whitespaces
sorted_data['Company Registered Number'] = sorted_data[
    'Company Registered Number'].apply(lambda x: str(x).strip())
sorted_data['Year'] = sorted_data['Year'].apply(lambda x: str(x).strip())
print("Cleaning Data")

# Drop unnecessary columns
sorted_data.drop(columns=['index'], inplace=True)
print("Dropping Unnecessary Columns")

# Drop duplicates if any
sorted_data.drop_duplicates(inplace=True)
print("Dropping duplicates")

# Sort Data frame by Year in descending order
sorted_data.groupby(
    ['Company Registered Number', 'Year']).head().sort_values(
    by=['Company Registered Number', "Year"], ascending=False, inplace=True)
# Resetting Index
sorted_data.reset_index(drop=True, inplace=True)
# Calculating % Change
sorted_data['Percentage Change'] = (sorted_data.groupby(
    ['Company Registered Number'])['Equity'].pct_change()*100)

# Isolating Equity and % Change by Year
BUFFER = []
for CN, group in tqdm(sorted_data.groupby(['Company Registered Number']), desc="Isolating % Change"):
    group.reset_index(drop=True, inplace=True)
    if len(group['Percentage Change']):
        for i in range(len(group['Percentage Change'])):
            group['PCT_'+group.iloc[i]['Year'].split("-")[0]] = group.iloc[i][
                'Percentage Change'].round(decimals=4)
            group['Equity_'+group.iloc[i]['Year'].split("-")[0]] = group.iloc[i][
                'Equity']
            BUFFER.append(group.iloc[i].to_dict())
    else:
        group[int(group.iloc[0]['Year'].split("-")[0]+1)] = 'No Data'
        BUFFER.append(group.to_dict())

# Function to fetch latest value among the given values
def latest(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]

# Percentage Change Dataframe
PCT_COLS = []
for column in pd.DataFrame(BUFFER).columns.to_list():
    if 'PCT' in column:
        PCT_COLS.append(column)
PCT_COLS.sort(reverse=True)

PCT_DF = pd.DataFrame(BUFFER)[
    ['Company Registered Number']+PCT_COLS]
PCT_DF.drop_duplicates(
    subset=['Company Registered Number'], keep='last', inplace=True)

PCT_DF.reset_index(drop=True, inplace=True)

# To file
PCT_DF.to_csv(OUT_FILE+'-PCTs.csv', index=False)

# Equity Data frame
EQUITY_COLS = []
for column in pd.DataFrame(BUFFER).columns.to_list():
    if 'Equity' in column:
        EQUITY_COLS.append(column)
EQUITY_COLS.sort(reverse=True)

EQUITY_DF = pd.DataFrame(BUFFER)[
    ['Company Registered Number']+EQUITY_COLS]
EQUITY_DF.drop_duplicates(
    subset=['Company Registered Number'], keep='last', inplace=True)

EQUITY_DF.reset_index(drop=True, inplace=True)

# To file
EQUITY_DF.to_csv(OUT_FILE+'-Equities.csv', index=False)

# Merging latest values
SEMI_FINAL_DF = pd.DataFrame(BUFFER)[[
    'Company Registered Number']].drop_duplicates().reset_index(drop=True)
tqdm.pandas(desc="Updating latest Equity")
SEMI_FINAL_DF['Equity'] = EQUITY_DF[EQUITY_COLS].apply(latest, axis=1)
tqdm.pandas(desc="Updating latest % Change")
SEMI_FINAL_DF['Percentage Change'] = PCT_DF[PCT_COLS].apply(latest, axis=1)

SEMI_FINAL_DF = SEMI_FINAL_DF.merge(
    Final_df[['Company Registered Number', 'SME Classification']],
    on=['Company Registered Number'])

# Equity Categorisation
l = ['< 0M', '[0M, 0.1M)', '[0.1M, 1M)', '[1M, 2M)',
     '[2M, 5M)', '[5M, 10M)', '[10M, 18M)', '> 18M']
SEMI_FINAL_DF['Equity Category'] = pd.cut(
    SEMI_FINAL_DF['Equity'],
    [ float('-inf'),
     0, 100000, 1000000, 2000000, 5000000, 10000000, 18000000,
      float('inf')], labels=l)
print("Equity wise Categorization done!")
# PCT Change Categorization
PCT_CATS = ['< -100', '[-100, -50)', '[-50, -25)', '[-25, -10)', '[-10,-2.5)',
            '[-2.5, 0)', '[0, 2.5)', '[2.5, 10)', '[10, 25)', '[25, 50)',
            '[50, 100)', '> 100', 'No Data']
if SEMI_FINAL_DF['Percentage Change'].min()-1 < -100 and SEMI_FINAL_DF[
        'Percentage Change'].max()+1 > 100:
    print(SEMI_FINAL_DF['Percentage Change'].min(),SEMI_FINAL_DF['Percentage Change'].max())
    SEMI_FINAL_DF['PCT Category'] = pd.cut(
        SEMI_FINAL_DF['Percentage Change'],
        [-150,
         -100, -50, -25, -10, -2.5, 0, 2.5, 10, 25, 50, 100, float(np.nan),
         SEMI_FINAL_DF['Percentage Change'].max()], labels=PCT_CATS)
elif SEMI_FINAL_DF['Percentage Change'].min()-1 < -100 and SEMI_FINAL_DF[
        'Percentage Change'].max()+1 < 100:
    SEMI_FINAL_DF['PCT Category'] = pd.cut(
        SEMI_FINAL_DF['Percentage Change'],
        [SEMI_FINAL_DF['Percentage Change'].min(),
         -100, -50, -25, -10, -2.5, 0, 2.5, 10, 25, 50, 100, float(np.nan)],
        labels=PCT_CATS[:-1])
elif SEMI_FINAL_DF['Percentage Change'].min()-1 > -100 and SEMI_FINAL_DF[
        'Percentage Change'].max()+1 > 100:
    SEMI_FINAL_DF['PCT Category'] = pd.cut(
        SEMI_FINAL_DF['Percentage Change'],
        [-100, -50, -25, -10, -2.5, 0, 2.5, 10, 25, 50, 100, float(np.nan),
         SEMI_FINAL_DF['Percentage Change'].max()], labels=PCT_CATS[1:])
else:
    SEMI_FINAL_DF['PCT Category'] = pd.cut(
        SEMI_FINAL_DF['Percentage Change'],
        [-100, -50, -25, -10, -2.5, 0, 2.5, 10, 25, 50, 100, float(np.nan)],
        labels=PCT_CATS[1:-1])

print("Percent wise categorisation done!")

# Replace inf or -inf if any
print("Repacing Infinite values..")
for i in tqdm(range(SEMI_FINAL_DF.first_valid_index(), SEMI_FINAL_DF.last_valid_index())):
    if SEMI_FINAL_DF['Percentage Change'].loc[i] == float('inf'):
        SEMI_FINAL_DF['Percentage Change'].loc[i] = 'Increase from Zero'
    elif SEMI_FINAL_DF['Percentage Change'].loc[i] == float('-inf'):
        SEMI_FINAL_DF['Percentage Change'].loc[i] == 'Decrease from Zero'
        
SEMI_FINAL_DF['Percentage Change'].fillna('No Data', inplace=True)
#SEMI_FINAL_DF['PCT Category'] = SEMI_FINAL_DF['PCT Category'].cat.add_categories('No Data')
SEMI_FINAL_DF['PCT Category'].fillna('No Data', inplace=True)
# Company Categorisation
SEMI_FINAL_DF.loc[
    SEMI_FINAL_DF['Equity Category'] == '< 0M', 'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0M, 0.1M)') &
                  (SEMI_FINAL_DF['PCT Category'].isin([
                      '> 100', '[50, 100)', '[25, 50)', '[10, 25)'])),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0M, 0.1M)') &
                  (~SEMI_FINAL_DF['PCT Category'].isin([
                      '> 100', '[50, 100)', '[25, 50)', '[10, 25)'])),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '> 100'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[50, 100)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[25, 50)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[10, 25)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[2.5, 10)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[0, 2.5)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-2.5, 0)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-10,-2.5)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-25, -10)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-50, -25)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-100, -50)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[0.1M, 1M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '< -100'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '> 100'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[50, 100)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[25, 50)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[10, 25)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[2.5, 10)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[0, 2.5)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-2.5, 0)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-10,-2.5)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-25, -10)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-50, -25)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-100, -50)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[1M, 2M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '< -100'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '> 100'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[50, 100)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[25, 50)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[10, 25)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[2.5, 10)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[0, 2.5)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-2.5, 0)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-10,-2.5)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-25, -10)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-50, -25)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-100, -50)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[2M, 5M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '< -100'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '> 100'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[50, 100)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[25, 50)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[10, 25)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[2.5, 10)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[0, 2.5)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-2.5, 0)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-10,-2.5)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-25, -10)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-50, -25)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-100, -50)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[5M, 10M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '< -100'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '> 100'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[50, 100)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[25, 50)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[10, 25)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[2.5, 10)'),
                  'Category'] = 'Champions'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[0, 2.5)'),
                  'Category'] = 'Contenders'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-2.5, 0)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-10,-2.5)'),
                  'Category'] = 'Prospects'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-25, -10)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-50, -25)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '[-100, -50)'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[(SEMI_FINAL_DF['Equity Category'] == '[10M, 18M)') &
                  (SEMI_FINAL_DF['PCT Category'] == '< -100'),
                  'Category'] = 'Strugglers'
SEMI_FINAL_DF.loc[SEMI_FINAL_DF['Equity Category'] == '> 18M',
                  'Category'] = 'Star Performers'
SEMI_FINAL_DF.loc[SEMI_FINAL_DF['PCT Category'] == 'No Data',
                  'Category'] = 'No Classification'

SEMI_FINAL_DF.to_csv(OUT_FILE+'-classified.csv', index=False)
print(f"Exported as {OUT_FILE+'-classified.csv'}")

END_TIME = time()
print(f"Time taken to run the script: {END_TIME-START_TIME} seconds")
