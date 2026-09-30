"""Python simulation and verification script for 4-to-2 Priority Encoder logic."""


def priority_encoder_4to2(enable: int, d3: int, d2: int, d1: int, d0: int) -> tuple[int, int]:
    """Computes output (Y1, Y0) for a 4-to-2 priority encoder.

    Priority order: D3 (highest) > D2 > D1 > D0 (lowest).
    When enable == 0, outputs are (0, 0).
    """
    if not enable:
        return 0, 0
    if d3:
        return 1, 1
    elif d2:
        return 1, 0
    elif d1:
        return 0, 1
    elif d0:
        return 0, 0
    return 0, 0


def generate_truth_table() -> list[dict]:
    """Generates complete truth table for all 2^5 = 32 input states."""
    rows = []
    for en in [0, 1]:
        for d3 in [0, 1]:
            for d2 in [0, 1]:
                for d1 in [0, 1]:
                    for d0 in [0, 1]:
                        y1, y0 = priority_encoder_4to2(en, d3, d2, d1, d0)
                        rows.append(
                            {
                                "enable": en,
                                "d3": d3,
                                "d2": d2,
                                "d1": d1,
                                "d0": d0,
                                "y1": y1,
                                "y0": y0,
                            }
                        )
    return rows


if __name__ == "__main__":
    table = generate_truth_table()
    print("Enable | D3 D2 D1 D0 | Y1 Y0")
    print("-" * 28)
    for r in table:
        if r["enable"]:
            print(
                f"   {r['enable']}   |  {r['d3']}  {r['d2']}  {r['d1']}  {r['d0']}  |  {r['y1']}  {r['y0']}"
            )
