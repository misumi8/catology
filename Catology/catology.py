import os
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import openpyxl
import matplotlib.pyplot as plt


def display_breed_counts(df, breed_column):
    breed_counts = df[breed_column].value_counts()
    print("Number of instances for each breed:")
    print(breed_counts)

def display_distinct_values_overall(df, breed_column):
    print("\nDistinct values and their frequencies for each attribute (overall):")
    for column in df.columns:
        if column != breed_column and column!='ID':
            print(f'\nDistinct values for {column}:')
            print(df[column].value_counts().sort_index(ascending=True))

            
def display_distinct_values_breed(df, breed_column):
    #with open(file_name, "w") as f:
        #f.write("\nDistinct values and their frequencies for each attribute (per breed):\n")
    for breed, group in df.groupby(breed_column):
        with open("distinctValuesBreeds/" + breed + ".txt", "w") as f:
            f.write(f'\nBreed: {breed}\n\nValue : Count\n')
            for column in df.columns:
                if column != breed_column and column != 'ID':
                    # f.write(f'Distinct values for {column} in {breed}:\n')
                    f.write(str(group[column].value_counts().sort_index(ascending=True)) + "\n\n")
    print(f"Output saved in the folder distinctValuesBreeds")
    os.startfile('distinctValuesBreeds')



# def analyze_breed_attributes(file_path):
#     df = pd.read_excel(file_path)
#     label_encoder = LabelEncoder()
#     df['Breed_encoded'] = label_encoder.fit_transform(df['Breed'])
#
#     X = df.drop(columns=['Breed', 'Breed_encoded', 'ID'])
#     y = df['Breed_encoded']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#
#     breed_mapping = dict(zip(df['Breed_encoded'], df['Breed']))
#
#     print("Classification report for each breed:")
#     print(classification_report(y_test, y_pred, target_names=[breed_mapping[b] for b in set(y)]))
#
#     print("\nFeature importance for predicting each breed:")
#     for breed_encoded in set(y):
#         breed_name = breed_mapping[breed_encoded]
#         print(f"\nBreed: {breed_name}")
#
#         breed_samples_idx = (y_test == breed_encoded)
#
#         if breed_samples_idx.sum() > 0:
#             importances_for_breed = model.feature_importances_
#
#             feature_importance = pd.DataFrame({
#                 'Feature': X.columns,
#                 'Importance': importances_for_breed
#             }).sort_values(by='Importance', ascending=False)
#
#             print(feature_importance.head(5))  # Show top 5 most important attributes for this breed


def compute_feature_importance_for_breeds(df):

    le = LabelEncoder()
    df['breed_encoded'] = le.fit_transform(df['Breed'])

    X = df.drop(columns=['Breed', 'breed_encoded','ID'])

    breed_feature_importance = {}

    for breed in df['breed_encoded'].unique():
        y = (df['breed_encoded'] == breed).astype(int)  # 1 for current breed, 0 for others

        clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
        # clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)

        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        breed_name = le.inverse_transform([breed])[0]
        breed_feature_importance[breed_name] = feature_importance

    return breed_feature_importance

#########################################################################################

def check_dataset_errors(df, expected_values=None):
    errors = {}

    missing_values = df.isnull().sum()
    missing_report = missing_values[missing_values > 0]
    if not missing_report.empty:
        errors['missing_values'] = missing_report
    else:
        errors['missing_values'] = "No missing values found"

    if expected_values:
        unexpected_values = {}
        for column, expected in expected_values.items():
            if column in df.columns:
                unexpected = df[~df[column].isin(expected)][column]
                if not unexpected.empty:
                    unexpected_values[column] = unexpected.value_counts()
        if unexpected_values:
            errors['unexpected_values'] = unexpected_values
        else:
            errors['unexpected_values'] = "No unexpected values found"
    else:
        errors['unexpected_values'] = "Expected values not provided for validation"

    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        errors['duplicates'] = duplicate_rows
    else:
        errors['duplicates'] = "No duplicate instances found"
    return errors


expected_values = {
    'Breed': ['Bengal', 'Birman LH', 'British Shorthair', 'Chartreux', 'European Shorthair',
              'Maine Coon', 'Persian', 'Ragdoll', 'Savannah', 'Sphynx', 'Turkish Van'],
    'Ext': [0,1,2,3,4,5],
    'Obs': [0,1,2,3,4,5],
    'Calm': [0,1,2,3,4,5],
    'Afraid': [0,1,2,3,4,5],
    'Intelligent': [0,1,2,3,4,5],
    'Vigilant': [0,1,2,3,4,5],
    'Persevering': [0,1,2,3,4,5],
    'Affectionate': [0,1,2,3,4,5],
    'Friendly': [0,1,2,3,4,5],
    'Solitary': [0,1,2,3,4,5],
    'Brutal': [0,1,2,3,4,5],
    'Dominant': [0,1,2,3,4,5],
    'Aggressive': [0,1,2,3,4,5],
    'Impulsive': [0,1,2,3,4,5],
    'Predictable': [0,1,2,3,4,5],
    'Distracted': [0,1,2,3,4,5],
    'PredBird': [0,1,2,3,4,5],
}

# ---------------------------------------------------------------------------------------------------------

def show_hist(fileName):
    df = pd.read_excel(fileName, engine='openpyxl')
    breeds = df['Breed'].unique()
    for breed in breeds:
        breed_data = df[df['Breed'] == breed]
        attributes = breed_data.drop(columns = ['Breed', 'ID'])
        plt.figure()
        attributes.mean().plot(kind='bar')
        plt.tight_layout()
        # df.hist(figsize=(15, 15))
        plt.subplots_adjust(left=0.073, right=0.718, top=0.935, bottom=0.324, wspace=0.255, hspace=0.442)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.title(f"{breed}")
        plt.xticks(rotation=45)
    plt.show()

def show_boxplot(fileName):
    df = pd.read_excel(fileName, engine='openpyxl')
    breeds = df['Breed'].unique()
    attributes = df.drop(columns=["ID", "Breed"]).columns
    for attribute in attributes:
        data_to_plot = [df[df['Breed'] == breed][attribute] for breed in breeds]
        plt.figure()
        plt.tight_layout()
        plt.boxplot(data_to_plot, labels=breeds, patch_artist=True)
        plt.title(f"'{attribute}'")
        plt.get_current_fig_manager().window.state('zoomed')
        plt.subplots_adjust(left=0.042, right=0.76, top=0.952, bottom=0.326, wspace=0.255, hspace=0.442)
        plt.xticks(rotation=45)
    plt.show()

def categorize(save = True):
    df = pd.read_excel('main.xlsx', engine='openpyxl')
    df_categorized = df[["ID", "Breed"]]
    for attribute_name in df.columns:
        if attribute_name not in {"ID", "Breed"}:
            attribute = df[[attribute_name]]
            enc = OneHotEncoder(sparse_output=False).fit(attribute)
            transformed_attribute = pd.DataFrame(enc.transform(attribute),
                                           columns=[attribute_name + str(cat) for cat in enc.categories_[0]])
            df_categorized = pd.concat([df_categorized, transformed_attribute], axis=1)
    if(save):
        df_categorized.to_excel('mainCategorized.xlsx', index=False)
        os.startfile('mainCategorized.xlsx')
    return df_categorized

def numerical_transform(to_categorize = False, save = True):
    if(to_categorize):
        df = categorize(False)
    else:
        df = pd.read_excel('main.xlsx', engine='openpyxl')
    le = LabelEncoder()
    df['Breed'] = le.fit_transform(df['Breed'])
    df.rename(columns={'Breed': 'Breed ID'}, inplace=True)
    if(save):
        df.to_excel('mainNumsOnly.xlsx', index=False)
        os.startfile('mainNumsOnly.xlsx')
    return df

# ---------------------------------------------------------------------------------------------------------

def interpret_errors(errors_report, df):
    suggestions = {}
    if 'missing_values' in errors_report:
        missing_values = errors_report['missing_values']
        if isinstance(missing_values, str):
            suggestions['missing_values'] = "No missing values, dataset looks complete in this regard."
        else:
            suggestions['missing_values'] = "Missing values found. Suggested approach and locations per column:\n"
            for column in missing_values.index:
                if column in df.columns[2:]:
                    missing_rows = df[df[column].isnull()].index.tolist()
                    for row in missing_rows:
                        breed = df.loc[row, 'Breed']
                        same_breed_rows = df[df['Breed'] == breed]
                        mean_value = same_breed_rows[column].mean()
                        if not pd.isna(mean_value):
                            suggested_value = int(round(mean_value))
                        else:
                            suggested_value = "No available data from the same breed to suggest a value."
                        suggestions['missing_values'] += (
                            f"- Column '{column}' has {missing_values[column]} missing values at row {row + 2}. "
                            f"Suggested value based on breed '{breed}': {suggested_value} (mean).\n"
                        )
    if 'unexpected_values' in errors_report:
        unexpected_values = errors_report['unexpected_values']
        if isinstance(unexpected_values, str):
            suggestions['unexpected_values'] = "No unexpected values found in the dataset."
        else:
            suggestions['unexpected_values'] = "Unexpected values found. Suggested approach and locations per column:\n"
            for column, unexpected in unexpected_values.items():
                unexpected_rows = df[~df[column].isin(expected_values[column])].index.tolist()
                adjusted_unexpected_rows = [row + 2 for row in unexpected_rows]
                suggestions['unexpected_values'] += f"- Column '{column}' has unexpected values at rows {adjusted_unexpected_rows}. Values: {list(unexpected.index)}. Suggested fix: Investigate and correct.\n"
    if 'duplicates' in errors_report:
        duplicates = errors_report['duplicates']
        if isinstance(duplicates, str):
            suggestions['duplicates'] = "No duplicate rows found."
        else:
            duplicate_rows = duplicates.index.tolist()
            adjusted_duplicate_rows = [row + 2 for row in duplicate_rows]
            suggestions['duplicates'] = f"Found {len(duplicates)} duplicate rows at rows {adjusted_duplicate_rows}. Suggested fix: Review and possibly remove duplicates."
    return suggestions


