import pandas as pd
import glob
import os

def merge_csvs_to_parquet(input_path, output_file, csv_naming_conv, chunk_size=100000):

    csv_files = glob.glob(input_path)
    
    if not csv_files:
        if csv_naming_conv:
            print(f"No CSV files found with {csv_naming_conv} in {input_path}")
            return False
        else:
            print(f"No CSV files found in {input_path}")
            return False           
        
    print(f"Found {len(csv_files)} CSV files")
   
    dfs = []
    
    for file in csv_files:
        try:
            if os.path.getsize(file) > 1e8:  
                chunks = pd.read_csv(file, chunksize=chunk_size)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file)
                
            dfs.append(df)
            print(f"Successfully read {file}")
            
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            continue
    
    if not dfs:
        print("No data frames to merge")
        return False
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    merged_df.to_parquet(output_file, index=False)
    print(f"Successfully created Parquet file: {output_file}")
    print(f"Final shape: {merged_df.shape}")
    
    return True
    

if __name__ == "__main__":
    
    csv_naming_conv = "anom_df_TRAIN_all_sats"
    input_path = f"/homes/dkurtenb/projects/russat/output/{csv_naming_conv}*.csv"  
    output_file = f"/homes/dkurtenb/projects/russat/output/{csv_naming_conv}_merged_data.parquet"  
        
    success = merge_csvs_to_parquet(input_path, output_file, csv_naming_conv)
    if success:
        print("Merge completed successfully")
    else:
        print("Merge failed")