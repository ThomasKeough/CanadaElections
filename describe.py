from clean import clean_dataset
import pandas as pd
import matplotlib.pyplot as plt

C = "mediumseagreen"
EC = "black"

df = clean_dataset("data/federal-candidates-2021-10-20.csv")

# find occupation distributions
occupations = list(df.columns)[15:]
sum_occupations = [df[col].sum() for col in occupations]
occupations_bars = pd.Series(sum_occupations, index=['Trades', 'Sales', 'Manufacturing', 'Law/Education/Government', 'Entertainment', 'Natural Resources',
                                                     'Natural Sciences', 'Member of Parliament', 'Management', 'Health', 'Business'])

# show occupation distributions
fig1 = plt.figure(1)
occupations_bars.plot.barh(color=C, edgecolor=EC)
plt.title("Distribution of Occupations among Candidates")
plt.xlabel("Number of Candidates")
plt.ylabel("Occupation Category")
fig1.tight_layout()

# find party distributions
parties = ['party_lc', 'party_cn', 'party_ndp', 'party_gr', 'party_other']
sum_parties = [df[col].sum() for col in parties]
parties_bars = pd.Series(sum_parties, index=['Liberal', 'Conservative', 'New Democrat', 'Green', 'Other'])

# show party distributions
fig2 = plt.figure(2)
parties_bars.plot.barh(color=['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey'], edgecolor=EC)
plt.title("Distribution of Political Parties among Candidates")
plt.xlabel("Number of Candidates")
plt.ylabel("Political Party")
fig2.tight_layout()

# plot of descriptive stats for other vars
describe = df[['elected', 'male', 'incumbent', 'indig', 'lawyer', 'switcher', 'multiple_candidacy']].describe()

def main():
    plt.show()
    print(describe.drop(index=['min', '25%', '50%', '75%', 'max']))


if __name__ == "__main__":
    main()
