import pandas as pd

def reverse_csv_rows(input_file, output_file):
    """
    Loads a CSV file, reverses the rows (last row becomes the first), and saves the updated file.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the updated CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)
        print("Original DataFrame:")
        print(df.head())

        # Reverse the rows
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        print("\nDataFrame after reversing rows:")
        print(df_reversed.head())

        # Save the updated DataFrame back to a CSV
        df_reversed.to_csv(output_file, index=False)
        print(f"\nUpdated CSV saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_csv = "/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/GBPUSD_1H.csv"  # Replace with your input file path
reverse_csv_rows(input_csv, input_csv)
