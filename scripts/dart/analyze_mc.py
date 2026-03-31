import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_results(csv_path):
    df = pd.read_csv(csv_path)
    
    # Target NED (from rm_dart_mc.cu)
    target_x = 25.233 * np.cos(7.3 * np.pi / 180.0)
    target_y = 25.233 * np.sin(7.3 * np.pi / 180.0)
    target_z = 1.5
    
    # Calculate error distance
    df['err_x'] = df['X'] - target_x
    df['err_y'] = df['Y'] - target_y
    df['err_z'] = df['Z'] - target_z
    
    df['miss_dist'] = np.sqrt(df['err_x']**2 + df['err_y']**2 + df['err_z']**2)
    
    hit_threshold = 0.055 / 2.0 # 55mm diameter -> 27.5mm radius
    hits = df[df['miss_dist'] <= hit_threshold]
    hit_rate = len(hits) / len(df) * 100.0
    
    print(f"Total Simulations: {len(df)}")
    print(f"Mean Miss Distance: {df['miss_dist'].mean()*1000:.2f} mm")
    print(f"Max Miss Distance: {df['miss_dist'].max()*1000:.2f} mm")
    print(f"Standard Deviation: {df['miss_dist'].std()*1000:.2f} mm")
    print(f"Hit Rate (55mm target): {hit_rate:.2f}%")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df['err_y']*1000, df['err_z']*1000, alpha=0.5, s=2)
    
    # Target circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(hit_threshold*1000*np.cos(theta), hit_threshold*1000*np.sin(theta), 'r-', label='55mm Target')
    
    plt.xlabel('Lateral Error (mm)')
    plt.ylabel('Vertical Error (mm)')
    plt.title(f'Monte Carlo Impact Distribution (Guidance: {"Active" if df["Guidance"].any() else "Inactive"})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('output/images/mc_impact_distribution.png')
    print("Plot saved to output/images/mc_impact_distribution.png")

if __name__ == "__main__":
    # Check if we are running from root or scripts/dart/
    csv_path = "output/logs/rm_dart_mc_results.csv"
    if not os.path.exists(csv_path):
        csv_path = "../../output/logs/rm_dart_mc_results.csv"
    analyze_results(csv_path)
