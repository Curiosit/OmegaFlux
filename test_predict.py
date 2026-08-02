import numpy as np
from omegaflux.predict import predict_room

def main():
    print("=" * 78)
    print("OmegaFlux - Prediction Workflow Test (README Canonical Example)")
    print("=" * 78)

    # Canonical room from the README (5.0m x 4.2m room, CCW)
    room = [(0, 0), (5.0, 0), (5.0, 4.2), (0, 4.2)]
    
    # Canonical window specification
    windows = [dict(wall=0, centre=2.5, width=1.8, sill=0.9, height=1.5)]

    print("\nRunning room prediction...")
    field = predict_room(room, windows, grid=0.25)  # -> x, y, UDI_useful

    if field.size > 0:
        print(f"\n[OK] Prediction completed successfully!")
        print(f"Total sensor points evaluated: {len(field):,}")
        print(f"Mean predicted UDI_useful: {field[:, 2].mean():.1f}% over {len(field)} points")
        
        print("\nFirst 5 points [x, y, UDI_useful]:")
        for row in field[:5]:
            print(f"  x={row[0]:.2f}m, y={row[1]:.2f}m -> UDI_useful={row[2]:.2f}%")
    else:
        print("\n[ERROR] Prediction returned empty results.")

if __name__ == "__main__":
    main()