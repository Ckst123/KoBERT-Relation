import pandas as pd
from sklearn.model_selection import train_test_split




if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    data = data[['headline', 'relation']].dropna()
    data['labels'] = (data['relation'] == 'O').astype('int')

    train, test = train_test_split(data, shuffle=True, train_size=0.8)

    print(train)
    print(test)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)