import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_video

def generate_heatmap():
    # Load tracks
    try:
        with open('tracks.pkl', 'rb') as f:
            tracks = pickle.load(f)
    except FileNotFoundError:
        print("Error: tracks.pkl not found. Run main.py first.")
        return

    # Load first frame for background
    video_frames = read_video('input.mp4')
    if not video_frames:
        print("Error: Could not read input.mp4")
        return
    background = video_frames[0]
    height, width, _ = background.shape

    # Collect all player positions
    all_points = []
    for track_id, data in tracks.items():
        if 'history' in data:
            all_points.extend(data['history'])
    
    if not all_points:
        print("No position history found.")
        return

    points = np.array(all_points)
    x = points[:, 0]
    y = points[:, 1]

    # Generate Heatmap
    print("Generating heatmap...")
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # Display background
    # Convert BGR to RGB for matplotlib
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    ax.imshow(background_rgb)

    # Overlay KDE plot
    # levels=10, thresh=0.05, alpha=0.5
    sns.kdeplot(x=x, y=y, fill=True, alpha=0.5, cmap='inferno', levels=10, ax=ax)
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0) # Invert Y axis to match image coordinates
    ax.axis('off')
    
    # Save result
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('heatmap_output.png', bbox_inches='tight', pad_inches=0)
    print("Heatmap saved to heatmap_output.png")
    plt.close()

if __name__ == '__main__':
    generate_heatmap()
