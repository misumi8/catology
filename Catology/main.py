import pandas as pd
from catology import *

df = pd.read_excel('main.xlsx')
# df = pd.read_excel('wrongdata.xlsx')
breed_column = 'Breed'

while(True):
    commands = '''
    Commands:
    1. Display the number of instances for each breed
    2. Display distinct values of breeds
    3. Display distinct values overall
    4. Attribute and class correlation 
    5. Find errors in dataset (missing or unsupported values, identical instances)
    6. Show potential solutions for the discovered errors
    7. Histogram
    8. Boxplot
    9. Open the dataset (numerical values only)
    
    
    Enter the order number of the desired command: '''
    user_input = input(commands)
    if(user_input.isdigit()):
        command_num = int(user_input)
        if(command_num == 1):
            display_breed_counts(df, breed_column)
        elif(command_num == 2):
            display_distinct_values_breed(df, breed_column)
        elif(command_num == 3):
            display_distinct_values_overall(df, breed_column)
        elif(command_num == 4):
            feature_importance_per_breed = compute_feature_importance_for_breeds(df)
            for breed, importance in feature_importance_per_breed.items():
                print(f"Breed: {breed}")
                print(importance)
                print("\n")
        elif(command_num == 5):
            errors_report = check_dataset_errors(df, expected_values)
            for error_type, details in errors_report.items():
                print("\n----------------------------------------------------")
                print(f"{error_type.upper()}:")
                pprint(details)
                print("----------------------------------------------------")
        elif(command_num == 6):
            errors_report = check_dataset_errors(df, expected_values)
            suggestions_report = interpret_errors(errors_report, df)
            print("\nSuggested solutions for adapting and completing the dataset:")
            for suggestion_type, suggestions in suggestions_report.items():
                print(f"\n{suggestion_type.upper()}:")
                print(suggestions)
        elif(command_num == 7):
            show_hist("main.xlsx")
        elif(command_num == 8):
            show_boxplot("main.xlsx")
        elif (command_num == 9):
            categorize = input("Transform categorical variables? [Y/N] ")
            if(categorize.upper() == "Y"):
                numerical_transform(True)
            elif(categorize.upper() == "N"):
                numerical_transform()
        else:
            print("[Error] Nonexistent command")
    elif(user_input == "exit"):
        break
