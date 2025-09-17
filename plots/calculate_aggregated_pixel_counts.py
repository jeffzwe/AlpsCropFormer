import os

aggregation_mapping = {
            # 0. Background / Non-crop
            0: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0,
            # 1. Meadow / Pasture
            36: 1, 37: 1,
            # 2. Soft Winter Wheat
            7: 2, 8: 2, 9: 2,
            # 3. Corn (Maize)
            6: 3,
            # 4. Winter Barley
            2: 4,
            # 5. Winter Rapeseed
            16: 5,
            # 6. Spring Barley
            1: 6,
            # 7. Sunflower
            18: 7,
            # 8. Grapevine
            39: 8, 43: 8,
            # 9. Beet
            12: 9, 13: 9,
            # 10. Winter Triticale
            4: 10,
            # 12. Fruits, Vegetables, Flowers
            24: 11, 25: 11, 27: 11, 28: 11,
            # 13. Potatoes
            14: 12,
            # 14. Leguminous Fodder
            21: 13, 22: 13, 23: 13, 34: 13, 38: 13,
            # 15. Soybeans
            17: 14,
            # 16. Orchard
            40: 15, 41: 15, 42: 15, 44: 15, 45: 15,
            # 17. Berries
            30: 16,
            # 18. Mixed Cereal
            3: 17, 5: 17, 10: 17, 11: 17, 29: 17,
            # 19. Sorghum
            26: 18,
            # 20. Other Oilseeds
            15: 19, 19: 19, 20: 19, 35: 19,
            # 21. Special Non-field / Perennial
            32: 20, 33: 20,
            # 22. Void label
            31: 21,
        }

def parse_pixel_counts(file_path):
    """Parse pixel counts from the text file."""
    pixel_counts = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(': ')
                class_id = int(parts[0].split(' ')[1])
                pixel_count = float(parts[1].split(' ')[0])
                pixel_counts[class_id] = pixel_count
    return pixel_counts

def calculate_aggregated_counts(original_counts, mapping):
    """Calculate aggregated pixel counts based on the mapping."""
    aggregated_counts = {}
    
    for original_class, aggregated_class in mapping.items():
        if original_class in original_counts:
            if aggregated_class not in aggregated_counts:
                aggregated_counts[aggregated_class] = 0
            aggregated_counts[aggregated_class] += original_counts[original_class]
    
    return aggregated_counts

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'class_pixel_count.txt')
    
    # Parse original pixel counts
    pixel_counts = parse_pixel_counts(file_path)
    
    # Calculate aggregated counts
    aggregated_counts = calculate_aggregated_counts(pixel_counts, aggregation_mapping)
    
    # Sort by class ID and display results
    print("Aggregated Class Pixel Counts:")
    print("=" * 40)
    for class_id in sorted(aggregated_counts.keys()):
        count = aggregated_counts[class_id]
        print(f"Aggregated Class {class_id}: {count:,.0f} pixels")
    
    # Calculate total pixels
    total_pixels = sum(aggregated_counts.values())
    print(f"\nTotal pixels: {total_pixels:,.0f}")
    
    # Calculate percentages
    print("\nClass Distribution:")
    print("=" * 40)
    for class_id in sorted(aggregated_counts.keys()):
        count = aggregated_counts[class_id]
        percentage = (count / total_pixels) * 100
        print(f"Class {class_id}: {percentage:.2f}%")

if __name__ == "__main__":
    main()
