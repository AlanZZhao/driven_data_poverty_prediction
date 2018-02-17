import pandas as pd

def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)

    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]
