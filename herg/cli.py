import argparse, sys
from herg.encoder_ext import encode
from herg.memory import MemoryCapsule
from herg import config

def demo_text():
    print("== HERG text demo ==")
    seed = "hello"
    vec, _ = encode(seed)
    cap = MemoryCapsule(1, vec, {})
    print(f"encoded '{seed}' → first 8 coords: {vec[:8].tolist()}")

def main(argv=None):
    cfg = config.load()
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["text"], default="text")
    parser.add_argument("--alpha", default=None,
                        help="comma-separated floats for kernel scale")
    parser.add_argument("--kernel", choices=["separable", "radial"],
                        default=None)
    args = parser.parse_args(argv)

    if args.alpha is not None:
        vals = [float(v) for v in args.alpha.split(',')] if ',' in args.alpha else float(args.alpha)
        cfg.kernel_alpha = vals
    if args.kernel is not None:
        cfg.kernel_mode = args.kernel
    config.atomic_save(cfg)

    if args.demo == "text":
        demo_text()

if __name__ == "__main__":
    main()
