"""
This file contains the build_attacker() factory function that dispatches
CHM-related attack names to their corresponding classes in CHM_Attackforloconet.py.

Usage in main.py:
    from method.CHM_dispatch import build_attacker
    attacker = build_attacker(args.attack, device=device, **vars(args))
"""

from method.CHM_Attackforloconet import CHM, CHM_HIDE, CHM_Plus, CAV_FINAL


def build_attacker(name: str, device: str, **kw):
    common = dict(device=device)

    # CHM_PLUS specific parameters
    for k in ['beta_cbr', 'p_rcmo', 'p_aeca', 'K1', 'beta_lra', 'beta_precr', 'gamma_vwcr']:
        if k in kw:
            common[k] = kw[k]

    # 名称规范化
    key = name.strip().upper()

    if key == 'CHM_FINAL':
        try:
            return CHM(**common)
        except TypeError:
            return CHM(device=device)

    raise ValueError(f"Unknown CHM attack method: {name}. "
                     f"Supported: CHM_FINAL, CHM_HIDE, CHM_PLUS, CHM_PLUS_LRA, CAV_FINAL")
