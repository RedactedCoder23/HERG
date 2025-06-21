import argparse, sys
from herg.encoder_ext import encode
from herg.memory import MemoryCapsule

def demo_text():
    print("== HERG text demo ==")
    seed = "hello"
    vec, _ = encode(seed)
    cap = MemoryCapsule(1, vec, {})
    print(f"encoded '{seed}' → first 8 coords: {vec[:8].tolist()}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["text"], default="text")
    args = parser.parse_args(argv)

    if args.demo == "text":
        demo_text()

if __name__ == "__main__":
    main()
