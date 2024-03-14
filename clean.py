import pandas as pd
import json

def clean_dataset(path: str):

    with open(path) as f:
        df = pd.read_csv(f, dtype={'country_birth': str,
                                   'lgbtq2_out': str})

    df = df.drop(columns={'party_minor_group','party_major_group','gov_minor_group','gov_major_group'})
    df['edate'] = df['edate'].apply(lambda x: x[:x.find('-')]).astype(float)

    # note 'id' is unique to a candidate across elections. 'id' will be repeated
    # if a candidate runs in mulitple elections

    # filter on 21st century
    df = df[df['edate'] > 2000]

    # build dummies from categorical vars
    df['elected'] = pd.get_dummies(df['elected'])['Elected']
    df['incumbent'] = pd.get_dummies(df['incumbent'])['Incumbent']
    df['male'] = pd.get_dummies(df['gender'])['M']
    df['indig'] = pd.get_dummies(df['indigenousorigins'])['Indigenous']
    df['multiple_candidacy'] = pd.get_dummies(df['multiple_candidacy'])['Multiple']
    df['lawyer'] = pd.get_dummies(df['lawyer'])['Lawyer']
    df['switcher'] = pd.get_dummies(df['switcher'])['Switcher']
    occupation_dummies = pd.get_dummies(df['censuscategory'])
    df = pd.concat([df, occupation_dummies], axis=1)

    
    # clean parties
    party_dummies = pd.get_dummies(df['party_raw'])
    df['party_lc'] = party_dummies['Liberal Party of Canada']
    df['party_cn'] = party_dummies['Conservative Party of Canada']
    df['party_ndp'] = party_dummies['New Democratic Party']
    df['party_gr'] = party_dummies['Green Party of Canada']
    df['party_other'] = 1
    df['party_other'].loc[df['party_lc'] == 1] = 0
    df['party_other'].loc[df['party_cn'] == 1] = 0
    df['party_other'].loc[df['party_ndp'] == 1] = 0
    df['party_other'].loc[df['party_gr'] == 1] = 0
    # df = df.drop('party_raw')

    # map strings to integers for gov_party_raw
    gov_parties = {"Conservative Party of Canada": 0,
                   "Liberal Party of Canada": 1}
    # apply mappings to string columns in df
    df['gov_party_raw'] = df['gov_party_raw'].replace(gov_parties)

    # clean census category column names
    new_census = {'Business, finance and administration occupations' : 'occ_business',
                  'Health occupations': 'occ_health',
                  'Management occupations': 'occ_manage',
                  'Members of Parliament': 'occ_mp',
                  'Natural and applied sciences and related occupations': 'occ_natsci',
                  'Natural resources, agriculture and related production occupations': 'occ_natres',
                  'Occupations in art, culture, recreation and sport': 'occ_entertain',
                  'Occupations in education, law and social, community and government services': 'occ_law_educ_gov',
                  'Occupations in manufacturing and utilities': 'occ_manufac',
                  'Sales and service occupations': 'occ_sales',
                  'Trades, transport and equipment operators and related occupations': 'occ_trades'}

    df = df.rename(columns=new_census)


    df.drop(columns=['gender', 'indigenousorigins'], inplace=True)
    df['age_at_elec'] = df['edate'] - df['birth_year']

    return df[['id', 'candidate_name', 'elected', 'male', 'incumbent',
               'indig', 'lawyer', 'switcher', 'multiple_candidacy',
               'gov_party_raw', 'party_lc', 'party_cn', 'party_ndp', 'party_gr', 'party_other'] + list(new_census.values())]
    

if __name__ == '__main__':
    df = clean_dataset('data/federal-candidates-2021-10-20.csv')
