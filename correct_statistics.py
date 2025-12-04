#!/usr/bin/env python3
"""
Script to correct the Wasserstein loss sign in training statistics.

The logged w_loss represents the Wasserstein distance (what discriminator maximizes),
but discriminator_loss should represent what the discriminator minimizes.

This script corrects the statistics so that:
- wasserstein_loss = E[D(real)] - E[D(fake)] (unchanged, this is the distance)
- discriminator_loss = -wasserstein_loss + gradient_penalty (corrected to show minimization objective)
"""

import csv
import argparse
import os
import shutil
from datetime import datetime


def correct_statistics(input_file, output_file=None, backup=True):
    """
    Correct the discriminator loss in training statistics.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save corrected file (if None, overwrites input)
        backup: Whether to create a backup of the original file
    """
    
    if output_file is None:
        output_file = input_file
        
    # Create backup if requested
    if backup and input_file == output_file:
        backup_file = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(input_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Read and process the data
    rows = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Correct discriminator loss: should be -w_loss + grad_penalty
            # Currently it's stored as w_loss, which is E[D(real)] - E[D(fake)]
            w_loss = float(row['wasserstein_loss'])
            grad_penalty = float(row['gradient_penalty'])
            
            # The true discriminator loss being minimized is:
            # -E[D(real)] + E[D(fake)] + lambda*GP = -w_loss + grad_penalty
            corrected_disc_loss = -w_loss + grad_penalty
            
            # Update the row
            row['discriminator_loss'] = f"{corrected_disc_loss:.6f}"
            rows.append(row)
    
    # Write corrected data
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Corrected {len(rows)} entries")
    print(f"Output saved to: {output_file}")
    
    # Show sample of corrections
    if rows:
        print("\nSample corrections (first 3 rows):")
        print(f"{'Iteration':<12} {'Old D Loss':<15} {'W Loss':<15} {'Grad Penalty':<15} {'New D Loss':<15}")
        print("-" * 72)
        
        # Re-read original for comparison
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            orig_rows = list(reader)
        
        for i in range(min(3, len(rows))):
            iteration = rows[i]['iteration']
            old_d = orig_rows[i]['discriminator_loss']
            w_loss = rows[i]['wasserstein_loss']
            grad_pen = rows[i]['gradient_penalty']
            new_d = rows[i]['discriminator_loss']
            print(f"{iteration:<12} {old_d:<15} {w_loss:<15} {grad_pen:<15} {new_d:<15}")


def main():
    parser = argparse.ArgumentParser(
        description='Correct discriminator loss sign in training statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Correct statistics in place (with backup)
  python correct_statistics.py training_statistics.csv
  
  # Correct and save to new file
  python correct_statistics.py input.csv -o corrected.csv
  
  # Correct without backup
  python correct_statistics.py training_statistics.csv --no-backup
        """
    )
    
    parser.add_argument('input_file', type=str, 
                       help='Path to input training_statistics.csv file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output file path (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup when overwriting')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        return 1
    
    # Perform correction
    try:
        correct_statistics(
            args.input_file, 
            args.output, 
            backup=not args.no_backup
        )
        print("\nCorrection completed successfully!")
        print("\nNote: The corrected discriminator_loss now represents the actual")
        print("minimization objective: -E[D(real)] + E[D(fake)] + gradient_penalty")
        return 0
        
    except Exception as e:
        print(f"Error during correction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
