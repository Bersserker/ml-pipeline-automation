"""Feature engineering. Implementation pending."""
import pandas as pd

def build_features(df: pd.DataFrame)-> pd.DataFrame:

    df =df.copy()

    df['se_ma_2'] = 0
    df.loc[((df.sex == 1) & (df.marriage == 1)) , 'se_ma_2'] = 1 #married man
    df.loc[((df.sex == 1) & (df.marriage == 2)) , 'se_ma_2'] = 2 #single man
    df.loc[((df.sex == 1) & (df.marriage == 3)) , 'se_ma_2'] = 3 #divorced man
    df.loc[((df.sex == 2) & (df.marriage == 1)) , 'se_ma_2'] = 4 #married woman
    df.loc[((df.sex == 2) & (df.marriage == 2)) , 'se_ma_2'] = 5 #single woman
    df.loc[((df.sex == 2) & (df.marriage == 3)) , 'se_ma_2'] = 6 #divorced woman

    df['agebin'] = 0 #creates a column of 0
    df.loc[((df['age'] > 20) & (df['age'] < 30)) , 'agebin'] = 1
    df.loc[((df['age'] >= 30) & (df['age'] < 40)) , 'agebin'] = 2
    df.loc[((df['age'] >= 40) & (df['age'] < 50)) , 'agebin'] = 3
    df.loc[((df['age'] >= 50) & (df['age'] < 60)) , 'agebin'] = 4
    df.loc[((df['age'] >= 60) & (df['age'] < 70)) , 'agebin'] = 5
    df.loc[((df['age'] >= 70) & (df['age'] < 81)) , 'agebin'] = 6

    df['avg_exp_5'] = ((df['bill_amt5'] - (df['bill_amt6'] - df['pay_amt5']))) / df['limit_bal']
    df['avg_exp_4'] = (((df['bill_amt5'] - (df['bill_amt6'] - df['pay_amt5'])) +
                    (df['bill_amt4'] - (df['bill_amt5'] - df['pay_amt4']))) / 2) / df['limit_bal']
    df['avg_exp_3'] = (((df['bill_amt5'] - (df['bill_amt6'] - df['pay_amt5'])) +
                    (df['bill_amt4'] - (df['bill_amt5'] - df['pay_amt4'])) +
                    (df['bill_amt3'] - (df['bill_amt4'] - df['pay_amt3']))) / 3) / df['limit_bal']
    df['avg_exp_2'] = (((df['bill_amt5'] - (df['bill_amt6'] - df['pay_amt5'])) +
                    (df['bill_amt4'] - (df['bill_amt5'] - df['pay_amt4'])) +
                    (df['bill_amt3'] - (df['bill_amt4'] - df['pay_amt3'])) +
                    (df['bill_amt2'] - (df['bill_amt3'] - df['pay_amt2']))) / 4) / df['limit_bal']
    df['avg_exp_1'] = (((df['bill_amt5'] - (df['bill_amt6'] - df['pay_amt5'])) +
                    (df['bill_amt4'] - (df['bill_amt5'] - df['pay_amt4'])) +
                    (df['bill_amt3'] - (df['bill_amt4'] - df['pay_amt3'])) +
                    (df['bill_amt2'] - (df['bill_amt3'] - df['pay_amt2'])) +
                    (df['bill_amt1'] - (df['bill_amt2'] - df['pay_amt1']))) / 5) / df['limit_bal']

    return df