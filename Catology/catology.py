import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def display_breed_counts(df, breed_column):
    breed_counts = df[breed_column].value_counts()
    print("Number of instances for each breed:")
    print(breed_counts)

def display_distinct_values_overall(df, breed_column):
    print("\nDistinct values and their frequencies for each attribute (overall):")
    for column in df.columns:
        if column != breed_column and column!='ID':
            print(f'\nDistinct values for {column}:')
            print(df[column].value_counts())

            
def display_distinct_values_breed(df, breed_column, file_name="distinct_values.txt"):
    with open(file_name, "w") as f:
        f.write("\nDistinct values and their frequencies for each attribute (per breed):\n")
        for breed, group in df.groupby(breed_column):
            f.write(f'\nBreed: {breed}\n')
            for column in df.columns:
                if column != breed_column and column != 'ID':
                    f.write(f'Distinct values for {column} in {breed}:\n')
                    f.write(str(group[column].value_counts()) + "\n")
    print(f"Output saved to {file_name}")



def analyze_breed_attributes(file_path):
    df = pd.read_excel(file_path)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Breed_encoded'] = label_encoder.fit_transform(df['Breed'])

    # Features and target variable
    X = df.drop(columns=['Breed', 'Breed_encoded', 'ID'])
    y = df['Breed_encoded']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get the predictions
    y_pred = model.predict(X_test)

    # Get the mapping of encoded breeds back to original breed names
    breed_mapping = dict(zip(df['Breed_encoded'], df['Breed']))

    # Print classification report
    print("Classification report for each breed:")
    print(classification_report(y_test, y_pred, target_names=[breed_mapping[b] for b in set(y)]))

    # For each predicted breed, let's analyze which attributes contributed the most
    print("\nFeature importance for predicting each breed:")
    for breed_encoded in set(y):
        breed_name = breed_mapping[breed_encoded]
        print(f"\nBreed: {breed_name}")

        # Select samples that belong to this breed
        breed_samples_idx = (y_test == breed_encoded)
        
        if breed_samples_idx.sum() > 0:
            # Get feature importances for this breed by taking the average of the importances for samples predicted as this breed
            importances_for_breed = model.feature_importances_
            
            # Sort and display the top features
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances_for_breed
            }).sort_values(by='Importance', ascending=False)

            print(feature_importance.head(5))  # Show top 5 most important attributes for this breed


def compute_feature_importance_for_breeds(df):

    # Encode the 'Breed' column into numerical values
    le = LabelEncoder()
    df['breed_encoded'] = le.fit_transform(df['Breed'])

    # Features (all columns except Breed and breed_encoded)
    X = df.drop(columns=['Breed', 'breed_encoded','ID'])

    # For each unique breed, compute the feature importance using a DecisionTreeClassifier
    breed_feature_importance = {}

    for breed in df['breed_encoded'].unique():
        # Filter the dataset for the current breed
        y = (df['breed_encoded'] == breed).astype(int)  # Binary classification: 1 for the current breed, 0 for others

        # Train a decision tree classifier for this breed
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Store the result
        breed_name = le.inverse_transform([breed])[0]
        breed_feature_importance[breed_name] = feature_importance

    return breed_feature_importance


df = pd.read_excel('wrongdata.xlsx')
breed_column = 'Breed'


#display_breed_counts(df, breed_column)
#display_distinct_values_breed(df, breed_column)
#display_distinct_values_overall(df,breed_column)

#Correlation
feature_importance_per_breed = compute_feature_importance_for_breeds(df)

'''for breed, importance in feature_importance_per_breed.items():
    print(f"Breed: {breed}")
    print(importance)
    print("\n")
    '''

#########################################################################################



def check_dataset_errors(df, expected_values=None):

    
    # Dictionary to hold error information
    errors = {}

    # 1. Check for missing values
    missing_values = df.isnull().sum()
    missing_report = missing_values[missing_values > 0]
    if not missing_report.empty:
        errors['missing_values'] = missing_report
    else:
        errors['missing_values'] = "No missing values found"

    # 2. Check for unexpected/extra values (if expected_values is provided)
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

    # 3. Check for duplicate instances
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        errors['duplicates'] = duplicate_rows
    else:
        errors['duplicates'] = "No duplicate instances found"

    # Return the error report
    return errors


expected_values = {
    'Breed': ['Bengal', 'Birman LH', 'British Shorthair', 'Chartreux', 'European Shorthair',
              'Maine Coon', 'Persian', 'Ragdoll', 'Savannah', 'Sphynx', 'Turkish Van'],
    # Add more columns and their expected values if necessary
}
errors_report = check_dataset_errors(df, expected_values)

# Display the errors report
for error_type, details in errors_report.items():
    print(f"\n{error_type.upper()}:")
    print(details)