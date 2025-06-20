"""Minimal gRPC server stubs for hvlogfs."""

import argparse
import time


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        print('gRPC server starting (dry-run)...')
        return
    # Actual server not implemented for tests
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
