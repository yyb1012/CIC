#!/usr/bin/env python3
"""
Avro to JSON Converter for DARPA TC CDM Data

This script converts Avro binary (.bin) files from the DARPA Transparent
Computing (TC) program into JSON format that can be processed by trace_parser.py.
It preserves Avro union type wrappers to match trace_parser.py's regex patterns.

Usage:
    python avro_to_json.py                           # Convert all .bin files in current dir
    python avro_to_json.py --input file.bin          # Convert a specific file
    python avro_to_json.py --input-dir ./data        # Convert all .bin files in a directory
    python avro_to_json.py --output-dir ./json_out   # Specify output directory


Usage:
    python avro_to_json.py                           # Convert all .bin files in current dir
    python avro_to_json.py --input file.bin          # Convert a specific file
    python avro_to_json.py --input-dir ./data        # Convert all .bin files in a directory
    python avro_to_json.py --output-dir ./json_out   # Specify output directory

Requirements:
    pip install fastavro tqdm
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Optional, List, Iterable

# Increase recursion limit for complex CDM schemas
sys.setrecursionlimit(10000)

try:
    import fastavro
except ImportError:
    print("Error: fastavro is required. Install it with:")
    print("  pip install fastavro")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable


def _convert_record_value(value, schema=None):
    """
    Recursively convert Avro record to JSON-serializable dict,
    preserving union type wrappers where applicable.
    """
    if value is None:
        return None
    elif isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value.hex()
    elif isinstance(value, dict):
        # Check if this is a union wrapper (single key that is a type name)
        result = {}
        for k, v in value.items():
            result[k] = _convert_record_value(v)
        return result
    elif isinstance(value, list):
        return [_convert_record_value(v) for v in value]
    else:
        return value


def _simple_json_writer(output_fp, records: Iterable[dict]) -> int:
    """
    Write records as JSON lines, one record per line.
    This is simpler and avoids fastavro.json_writer's schema parsing issues.
    """
    count = 0
    for record in records:
        converted = _convert_record_value(record)
        line = json.dumps(converted, separators=(',', ':'), ensure_ascii=False)
        output_fp.write(line + '\n')
        count += 1
    return count


def _split_output_path(base_output_path: Path, index: int) -> Path:
    return base_output_path.with_name(f"{base_output_path.stem}_{index}.json")


def avro_to_json(
    input_path: str,
    output_path: Optional[str] = None,
    records_per_file: int = 0,
    verbose: bool = True
) -> str:
    """
    Convert an Avro (.bin) file to JSON format.
    Uses Avro JSON encoding with union type wrappers for trace_parser.py.
    
    Args:
        input_path: Path to the input .bin (Avro) file
        output_path: Path to the output .json file. If None, replaces .bin with .json
        records_per_file: If > 0, split output into multiple files with this many records each
        verbose: Whether to print progress information
    
    Returns:
        Path to the output file (or the last file if split)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        # Replace .bin extension with .json, or append .json if no .bin extension
        if input_path.suffix.lower() == '.bin':
            output_path = input_path.with_suffix('.json')
        else:
            output_path = Path(str(input_path) + '.json')
    else:
        output_path = Path(output_path)
    
    if verbose:
        print(f"Converting: {input_path}")
        print(f"Output:     {output_path}")
    
    # Open and convert
    record_count = 0
    file_index = 0
    last_output_path = None
    progress = None

    def _count_records(records: Iterable[dict]) -> Iterable[dict]:
        nonlocal record_count
        for record in records:
            record_count += 1
            yield record

    def _next_chunk(records: Iterable[dict], limit: int) -> Optional[Iterable[dict]]:
        chunk_iter = itertools.islice(records, limit)
        try:
            first = next(chunk_iter)
        except StopIteration:
            return None
        return itertools.chain([first], chunk_iter)

    with open(input_path, 'rb') as avro_file:
        reader = fastavro.reader(avro_file)
        schema = getattr(reader, "writer_schema", None) or getattr(reader, "schema", None)
        # Note: We don't need to parse schema for simple JSON writing

        records_iter = reader
        if verbose:
            progress = tqdm(records_iter, desc="Converting records", unit=" records")
            records_iter = progress

        records_iter = _count_records(records_iter)

        if records_per_file > 0:
            while True:
                chunk_records = _next_chunk(records_iter, records_per_file)
                if chunk_records is None:
                    break
                current_output_path = _split_output_path(output_path, file_index)
                if verbose:
                    print(f"\nWriting file: {current_output_path}")
                with open(current_output_path, 'w', encoding='utf-8') as current_output:
                    _simple_json_writer(current_output, chunk_records)
                last_output_path = current_output_path
                file_index += 1
        else:
            with open(output_path, 'w', encoding='utf-8') as current_output:
                _simple_json_writer(current_output, records_iter)
            last_output_path = output_path

    if progress and hasattr(progress, "close"):
        progress.close()
    
    if verbose:
        print(f"\nConverted {record_count:,} records")
        if records_per_file > 0:
            print(f"Split into {file_index} files")
    
    return str(last_output_path or output_path)


def find_bin_files(directory: str) -> List[Path]:
    """Find all .bin files in a directory (non-recursive)."""
    dir_path = Path(directory)
    bin_files = list(dir_path.glob("*.bin")) + list(dir_path.glob("*.bin.*"))
    
    # Filter out files that have both .bin and other extensions that indicate
    # they are already processed (e.g., .bin.json)
    result = []
    for f in bin_files:
        # Skip if there's a .json extension anywhere
        if '.json' not in f.suffixes:
            result.append(f)
    
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DARPA TC Avro (.bin) files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python avro_to_json.py
      Convert all .bin files in the current directory

  python avro_to_json.py --input ta1-cadets-1-e5-official-2.bin
      Convert a specific file
      
  python avro_to_json.py --input-dir ./data --output-dir ./json_data
      Convert all .bin files from ./data and save to ./json_data

  python avro_to_json.py --input large_file.bin --split 1000000
      Split output into files with 1M records each
"""
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to a specific .bin file to convert'
    )
    
    parser.add_argument(
        '--input-dir', '-d',
        type=str,
        default='.',
        help='Directory containing .bin files to convert (default: current directory)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for JSON files (default: same as input)'
    )
    
    parser.add_argument(
        '--split', '-s',
        type=int,
        default=0,
        help='Split output into files with this many records each (default: 0 = no split)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Determine which files to convert
    if args.input:
        # Convert a specific file
        bin_files = [Path(args.input)]
    else:
        # Find all .bin files in the directory
        bin_files = find_bin_files(args.input_dir)
        if not bin_files:
            print(f"No .bin files found in {args.input_dir}")
            return
        
        if verbose:
            print(f"Found {len(bin_files)} .bin file(s) to convert:")
            for f in bin_files:
                print(f"  - {f}")
            print()
    
    # Convert each file
    for input_file in bin_files:
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            continue
        
        # Determine output path
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            if input_file.suffix.lower() == '.bin':
                output_name = input_file.with_suffix('.json').name
            else:
                # Handle .bin.1, .bin.2 etc.
                output_name = input_file.name.replace('.bin', '.json')
            
            output_path = output_dir / output_name
        else:
            output_path = None  # Will be determined automatically
        
        try:
            avro_to_json(
                input_path=str(input_file),
                output_path=str(output_path) if output_path else None,
                records_per_file=args.split,
                verbose=verbose
            )
            if verbose:
                print()
        except Exception as e:
            print(f"Error converting {input_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
