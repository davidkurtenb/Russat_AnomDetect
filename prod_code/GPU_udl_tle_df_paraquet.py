import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import torch
import pandas as pd
import numpy as np
from spacetrack import SpaceTrackClient
import requests
import base64
import pickle
from tqdm import tqdm

def setup(rank, world_size):
    """Initialize distributed computing environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed computing environment"""
    dist.destroy_process_group()

class DistributedTLEProcessor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        if rank == 0:
            print(f"Process {rank} using device: {self.device}")

    def process_batch(self, df_batch):
        """Process a batch of TLE data using assigned GPU"""
        if len(df_batch) == 0:
            return None
            
        numeric_cols = df_batch.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df_batch:
                tensor = torch.tensor(df_batch[col].values, dtype=torch.float32, device=self.device)
                df_batch[f'{col}_mean'] = tensor.mean().cpu().numpy()
                df_batch[f'{col}_std'] = tensor.std().cpu().numpy()
                
        return df_batch

def process_node(rank, world_size, cntry_nm, output_dir, st_un, st_pw, udl_un, udl_pw, batch_size):
    """Main processing function for each node/GPU"""
    # Initialize distributed environment
    setup(rank, world_size)
    
    # Initialize processor for this rank
    processor = DistributedTLEProcessor(rank, world_size)
    
    # Only rank 0 fetches the initial satellite catalog
    if rank == 0:
        st = SpaceTrackClient(identity=st_un, password=st_pw)
        rus_sat = st.satcat(country=cntry_nm, current='Y')
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{cntry_nm}_satcat.pkl'), 'wb') as file:
            pickle.dump(rus_sat, file)
            
        cis_norad_ids = [int(e.get('NORAD_CAT_ID')) for e in rus_sat]
    else:
        cis_norad_ids = None
    
    # Broadcast NORAD IDs to all processes
    cis_norad_ids = torch.tensor(cis_norad_ids, device='cpu') if rank == 0 else torch.empty(0, device='cpu')
    dist.broadcast(cis_norad_ids, src=0)
    
    # Distribute work among processes
    total_ids = len(cis_norad_ids)
    ids_per_process = total_ids // world_size
    start_idx = rank * ids_per_process
    end_idx = start_idx + ids_per_process if rank != world_size - 1 else total_ids
    
    # Process assigned NORAD IDs
    basicAuth = "Basic " + base64.b64encode(f"{udl_un}:{udl_pw}".encode('utf-8')).decode("ascii")
    current_batch = []
    output_file = os.path.join(output_dir, f'{cntry_nm}_tle_data_rank{rank}.parquet')
    
    for idx in tqdm(range(start_idx, end_idx), desc=f"Rank {rank} processing", disable=rank != 0):
        value = cis_norad_ids[idx].item()
        
        try:
            url = f"https://unifieddatalibrary.com/udl/elset/history?epoch=%3E2012-02-01T00:00:00.000000Z&satNo={value}"
            result = requests.get(url, headers={'Authorization':basicAuth}, verify=False)
            df = pd.DataFrame(result.json())
            
            if not df.empty:
                current_batch.append(df)
            
            if len(current_batch) >= batch_size or idx == end_idx - 1:
                if current_batch:
                    batch_df = pd.concat(current_batch, ignore_index=True)
                    processed_batch = processor.process_batch(batch_df)
                    
                    if processed_batch is not None:
                        mode = 'a' if os.path.exists(output_file) else 'w'
                        processed_batch.to_parquet(
                            output_file,
                            index=False,
                            engine='fastparquet',
                            mode=mode
                        )
                    
                    current_batch = []
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            if rank == 0:
                print(f"Error processing NORAD ID {value}: {str(e)}")
            continue
    
    # Wait for all processes to complete
    dist.barrier()
    
    # Combine results from all ranks (only on rank 0)
    if rank == 0:
        all_files = [os.path.join(output_dir, f'{cntry_nm}_tle_data_rank{r}.parquet') 
                    for r in range(world_size)]
        dfs = [pd.read_parquet(f) for f in all_files if os.path.exists(f)]
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(
                os.path.join(output_dir, f'{cntry_nm}_tle_data_combined.parquet'),
                index=False,
                engine='fastparquet'
            )
            
            # Clean up individual rank files
            for f in all_files:
                if os.path.exists(f):
                    os.remove(f)
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    output_dir = '/homes/dkurtenb/projects/russat/output'

    st_un = 'dk4120@gmail.com'      
    st_pw = 'Sup3rDup3r!98600'
    udl_un = 'david.kurtenbach2'
    udl_pw ='$up3rDup3r!98600'

    credentials = {
        'cntry_nm': 'CIS',
        'output_dir': output_dir,
        'st_un': st_un,
        'st_pw': st_pw,
        'udl_un': udl_un,
        'udl_pw': udl_pw,
        'batch_size': 100
    }
    
    mp.spawn(
        process_node,
        args=(world_size, *credentials.values()),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()