"""exp111 (docs/plan/blind_transfer.md, Pins, amendment of 16 October 2026): rslearn with v1's own temporal table.

    python -I -u exp/exp111_v1table.py --ckpt CKPT --record RECORD [--variant V] -- <rslearn arguments, e.g. model predict>

Ai2's FT-AWF v1 checkpoint stores the encoder's temporal position table
(model.encoder.0.model.composite_encodings.pos_embed, 12 x 192), trained along with the encoder in Ai2's fine-tune: it
differs from the sinusoidal formula by up to 0.0037, row 0 included, which no float format's rounding explains.
olmoearth-pretrain-minimal 0.1.2 (under rslearn 0.1.17) drops that table at load and computes the formula's at every
forward (flexi_vit's one call of get_1d_sincos_pos_encoding, in CompositeEncodings), and with it v1 failed gate A
(job 1365118: 245 of 259 the same class as exp98's). olmoearth_pretrain 0.1.0 added the stored table's first t rows
(exp98 ran 0.0.2, not inspected). This runs rslearn's own CLI unchanged except for that one call, which returns the
stored table's first t rows at the table's width; any other width falls through to the formula.

Variants (--variant), for the report-only diagnostic of 16 October 2026 (e111.sh mode v1diag) and as candidate routes:
  formula        rslearn as is (the record only; the route v1 failed gate A with);
  table          the stored table, as above;
  table_fp32buf  the stored table, and the encodings added in float32 as olmoearth_pretrain 0.1.0 built them
                 (CompositeEncodings' buffer was float32, so under bf16 autocast the tokens left it in float32; 0.1.2
                 builds it in the tokens' dtype): the tokens enter _apply_encodings_per_modality as float32;
  table_fp32     the stored table, and rslearn's OlmoEarth wrapper without its bfloat16 autocast (all float32).

The record (RECORD, keys checked as every exp111 record) holds the variant, the table's difference from the formula,
the counts of calls each patch answered, rslearn's exit (or the exception that ended it), the Slurm job and the commit.
It refuses (exit 3) when rslearn exited 0 but a patch the variant needs was never used, so a route that silently did
not apply cannot pass as this one.
"""
import argparse
import os
import sys

TABLE_KEY = "model.encoder.0.model.composite_encodings.pos_embed"
VARIANTS = ("formula", "table", "table_fp32buf", "table_fp32")


def install_own_table(table):
    """Make olmoearth's flexi_vit answer its temporal-encoding call from `table` (T x D): the first t rows when asked for
    width D and t <= T, an error when t > T (the checkpoint's code could not have run), the formula for any other width.
    Returns the call counts, updated in place."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import flexi_vit
    original = flexi_vit.get_1d_sincos_pos_encoding
    calls = {"own": 0, "formula": 0}

    def own(pos, encoding_dim):
        n = pos.reshape(-1).shape[0]
        if encoding_dim != table.shape[1]:
            calls["formula"] += 1
            return original(pos, encoding_dim)
        if n > table.shape[0]:
            raise ValueError(f"{n} timesteps, above the stored table's {table.shape[0]}")
        calls["own"] += 1
        return table[:n].to(device=pos.device)

    flexi_vit.get_1d_sincos_pos_encoding = own
    return calls


def install_fp32_encodings(calls):
    """Make CompositeEncodings add its encodings in float32 (olmoearth_pretrain 0.1.0's arithmetic): the tokens enter
    _apply_encodings_per_modality as float32, so 0.1.2's buffer, built in the tokens' dtype, is float32 too."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import flexi_vit
    original = flexi_vit.CompositeEncodings._apply_encodings_per_modality

    def fp32(self, modality_name, modality_tokens, *args, **kwargs):
        calls["fp32buf"] += 1
        return original(self, modality_name, modality_tokens.float(), *args, **kwargs)

    flexi_vit.CompositeEncodings._apply_encodings_per_modality = fp32


def install_no_autocast(calls):
    """Make rslearn's OlmoEarth wrapper run without autocast (autocast_dtype None after its own __init__)."""
    from rslearn.models.olmoearth_pretrain import model as oe_model
    original = oe_model.OlmoEarth.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.autocast_dtype = None
        calls["no_autocast"] += 1

    oe_model.OlmoEarth.__init__ = init


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: exp111_v1table.py --ckpt CKPT --record RECORD [--variant V] -- <rslearn arguments>")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="table")
    a = ap.parse_args(sys.argv[1:cut])
    rest = sys.argv[cut + 1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script's directory off sys.path
    import torch
    import exp111_jobs as jobs
    sd, how = jobs.load_checkpoint_safely(a.ckpt)
    if TABLE_KEY not in sd or not isinstance(sd[TABLE_KEY], torch.Tensor) or tuple(sd[TABLE_KEY].shape) != (12, 192):
        raise SystemExit(f"refused: the checkpoint holds no 12 x 192 {TABLE_KEY}")
    table = sd[TABLE_KEY].detach().to(torch.float32).clone()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import get_1d_sincos_pos_encoding
    formula = get_1d_sincos_pos_encoding(torch.arange(table.shape[0]), table.shape[1]).to(torch.float32)
    rec = {"variant": a.variant, "table_key": TABLE_KEY, "load": how, "table_shape": list(table.shape),
           "table_max_abs_diff_from_formula": float((table - formula).abs().max()),
           "table_row0_max_abs_diff": float((table[0] - formula[0]).abs().max()),
           "job": os.environ.get("SLURM_JOB_ID"), "commit": os.environ.get("E111_SHA")}
    own = install_own_table(table) if a.variant != "formula" else {"own": 0, "formula": 0}
    calls = {"fp32buf": 0, "no_autocast": 0}
    if a.variant == "table_fp32buf":
        install_fp32_encodings(calls)
    if a.variant == "table_fp32":
        install_no_autocast(calls)
    from rslearn.main import main as rslearn_main
    sys.argv = ["rslearn", *rest]
    code = 0
    try:
        rslearn_main()
    except SystemExit as ex:
        code = ex.code if isinstance(ex.code, int) else (0 if ex.code is None else 1)
    except BaseException as ex:
        code = f"exception: {type(ex).__name__}"
        raise
    finally:
        calls.update(own)
        rec.update({"calls_own_table": calls["own"], "calls_formula": calls["formula"],
                    "calls_fp32_encodings": calls["fp32buf"], "wrappers_without_autocast": calls["no_autocast"],
                    "rslearn_exit": code})
        jobs.write_json(rec, a.record)
    need = {"table": ("own",), "table_fp32buf": ("own", "fp32buf"), "table_fp32": ("own", "no_autocast")}
    unused = [k for k in need.get(a.variant, ()) if calls[k] == 0]
    if code == 0 and unused:
        print(f"refused: rslearn ran but the {a.variant} route never applied {unused}"
              + (" (v1's stored table was never used)" if "own" in unused else ""), file=sys.stderr)
        sys.exit(3)
    sys.exit(code)


if __name__ == "__main__":
    main()
