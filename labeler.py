import pandas as pd

UBER_DATA = "uber_cleaned.csv"
LYFT_DATA = "lyft_cleaned.csv"
SAVE_FREQUENCY = 10

dataset = input("Which dataset would you like to label? (uber/lyft) ")
data = None
if (dataset == "uber"):
    data = pd.read_csv(UBER_DATA)
elif (dataset == "lyft"):
    data = pd.read_csv(LYFT_DATA)

def print_important_row_info(row, num):
    print("\n")
    print(f"Index: {num}")
    print(f"Rating: {row.Rating}")
    print(f"Review Body: {row['Review Body']}")
    print(f"Company asked for review? {row['Invited']}")
    print("\n")
    

if isinstance(data, pd.DataFrame):
    print(f"The next unlabeled data is this row:\n{data[data.Unfair.isnull()].head(1)}")
    index = int(input("Which (integer) index are you starting from? "))
    while True:
        print_important_row_info(data.iloc[index], index)
        unfair = ""
        while unfair not in ['y', 'n']:
            unfair = input("Is this review unfair? (y/n) ")
        data.loc[index, "Unfair"] = unfair == 'y'
        index += 1

        if index % SAVE_FREQUENCY == 0:
            if dataset == 'uber':
                data.to_csv(UBER_DATA, index=False)
            else:
                data.to_csv(LYFT_DATA, index=False)
            print("Writing data to file.")
        
        print("\n")

    
