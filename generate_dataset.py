import pandas as pd
import numpy as np
import os

def generate_flight_data(filename='tiketcom_bestprice.csv'):
    np.random.seed(50)
    
    routes = [
        ('JKTC', 'YIA', 367200.0, 764280.0, 480000.0),
        ('JKTC', 'DPS', 550000.0, 1450000.0, 850000.0),
        ('JKTC', 'SUB', 420000.0, 980000.0, 620000.0),
        ('JKTC', 'KNO', 780000.0, 1850000.0, 1150000.0),
        ('JKTC', 'UPG', 820000.0, 1950000.0, 1250000.0)
    ]
    
    start_date = '2023-01-01'
    end_date = '2024-06-26'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    rows = []
    
    for origin, dest, min_p, max_p, base_p in routes:
        n_days = len(dates)
        
        # Base trend with sinus wave + seasonality + weekend bump
        t = np.arange(n_days)
        seasonal = np.sin(2 * np.pi * t / 365.25) * (base_p * 0.15)
        monthly = np.cos(2 * np.pi * t / 30) * (base_p * 0.05)
        
        # Ramadhan / Lebaran spikes (April 2023 and April 2024)
        lebaran_spike = np.exp(-((t - 110)**2) / 100.0) * (base_p * 0.45) + \
                        np.exp(-((t - 475)**2) / 100.0) * (base_p * 0.45)
                        
        # Year end spike
        year_end = np.exp(-((t - 360)**2) / 80.0) * (base_p * 0.35)
        
        noise = np.random.normal(0, base_p * 0.04, size=n_days)
        
        for i, d in enumerate(dates):
            dow = d.dayofweek
            weekend_factor = 1.12 if dow in [4, 5, 6] else 1.0  # Fri, Sat, Sun premium
            
            price = (base_p + seasonal[i] + monthly[i] + lebaran_spike[i] + year_end[i] + noise[i]) * weekend_factor
            price = np.clip(price, min_p, max_p)
            price = round(price / 100.0) * 100.0  # Round to hundreds
            
            # Generate 2-4 extract timestamps per depart_date to mimic scraping data
            n_entries = np.random.randint(2, 5)
            for _ in range(n_entries):
                rows.append({
                    'extract_timestamp': f"{d.strftime('%Y-%m-%d')} {np.random.randint(8,22):02d}:{np.random.randint(0,59):02d}:{np.random.randint(0,59):02d}.123456",
                    'origin': origin,
                    'destination': dest,
                    'depart_date': d.strftime('%Y-%m-%d'),
                    'best_price': float(price + np.random.choice([0, 5000, 10000, -5000]))
                })

    df = pd.DataFrame(rows)
    df.to_csv(filename, sep='|', index=False)
    print(f"Dataset berhasil dibuat: {filename} ({len(df)} baris)")

if __name__ == '__main__':
    generate_flight_data()
