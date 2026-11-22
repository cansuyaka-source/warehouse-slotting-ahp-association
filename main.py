# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
import pandas as pd

# 1. Tezdeki doğrulanmış AHP kriter ağırlıkları (CR = 0.0204 < 0.10)[cite: 16, 17]
AHP_WEIGHTS = {
    "order_freq": 0.3999,        # Sipariş sıklığı[cite: 16]
    "joint_order_ratio": 0.2427, # Birlikte sipariş edilme durumu[cite: 16]
    "triple_assoc": 0.1592,      # 3'lü ürün ilişkisi[cite: 16]
    "pair_assoc": 0.1200,        # 2'li ürün ilişkisi[cite: 16]
    "total_qty": 0.0783,         # Miktar[cite: 16]
}

# 2. Tezdeki senaryo mesafe karşılaştırma sonuçları[cite: 29, 36]
THESIS_RESULTS = {
    "alt1_toplam_mesafe": 9217852.95,
    "alt2_toplam_mesafe": 7506817.05,
    "alt1_birliktelik_mesafe": 6287942.75,
    "alt2_birliktelik_mesafe": 4253570.20,
    "alt1_giris_mesafe": 2929910.20,
    "alt2_giris_mesafe": 3253246.85,
}


def generate_synthetic_orders(n_orders: int = 600, seed: int = 42) -> pd.DataFrame:
    """Gerçek şirket verisi yerine örnek sipariş satırları üretir."""
    rng = np.random.default_rng(seed)
    skus = [f"SKU_{i:03d}" for i in range(1, 51)]
    weights = np.exp(-np.linspace(0, 3.0, len(skus)))
    weights = weights / weights.sum()

    records = []
    for idx in range(1, n_orders + 1):
        delivery_id = f"DEL_{idx:05d}"
        r = rng.random()
        if r < 0.25:
            basket = ["SKU_001", "SKU_002", "SKU_003"]
        elif r < 0.55:
            basket = ["SKU_001", "SKU_004"]
        else:
            k = int(rng.choice([1, 2, 3, 4], p=[0.30, 0.38, 0.20, 0.12]))
            basket = list(rng.choice(skus, size=k, replace=False, p=weights))

        for sku in basket:
            records.append({
                "delivery": delivery_id,
                "material": sku,
                "order_qty": int(rng.integers(1, 30)),
                "gross_weight_g": float(rng.uniform(150, 2400)),
                "desi": float(rng.uniform(0.10, 2.20)),
            })

    return pd.DataFrame(records)


def minmax_scale(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)


def run_analysis(orders_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pareto/ABC, AHP skorlama ve 2 katlı raf (25x30x2) atamasını çalıştırır[cite: 8, 16, 20]."""
    sku_df = (
        orders_df.groupby("material", as_index=False)
        .agg(
            order_freq=("delivery", "nunique"),
            total_qty=("order_qty", "sum"),
            avg_weight_g=("gross_weight_g", "mean"),
            avg_desi=("desi", "mean"),
        )
    )

    basket_sizes = orders_df.groupby("delivery")["material"].transform("count")
    joint_ratio = (
        orders_df.assign(is_joint=(basket_sizes > 1).astype(int))
        .groupby("material", as_index=False)["is_joint"]
        .mean()
        .rename(columns={"is_joint": "joint_order_ratio"})
    )

    df = sku_df.merge(joint_ratio, on="material", how="left")
    df = df.sort_values("order_freq", ascending=False).reset_index(drop=True)

    # Pareto kümülatif pay ve ABC sınıflandırması[cite: 8]
    df["cum_share"] = df["order_freq"].cumsum() / df["order_freq"].sum()
    df["abc_class"] = df["cum_share"].apply(
        lambda x: "A" if x <= 0.80 else ("B" if x <= 0.95 else "C")
    )

    # Örnek veri için birliktelik skoru (gerçek veride SQL ikili/üçlü çıktısı bağlanır)[cite: 13]
    df["pair_assoc"] = df["joint_order_ratio"] * df["order_freq"]
    df["triple_assoc"] = (df["joint_order_ratio"] ** 2) * df["order_freq"]

    # AHP skoru[cite: 16]
    df["ahp_score"] = (
        AHP_WEIGHTS["order_freq"] * minmax_scale(df["order_freq"])
        + AHP_WEIGHTS["joint_order_ratio"] * minmax_scale(df["joint_order_ratio"])
        + AHP_WEIGHTS["triple_assoc"] * minmax_scale(df["triple_assoc"])
        + AHP_WEIGHTS["pair_assoc"] * minmax_scale(df["pair_assoc"])
        + AHP_WEIGHTS["total_qty"] * minmax_scale(df["total_qty"])
    )
    df = df.sort_values("ahp_score", ascending=False).reset_index(drop=True)

    # 25 satır x 30 sütun x 2 kat raf ızgarası (Giriş: 1,1 | Üst kat ceza: 0.15)[cite: 20, 22]
    slots = []
    for level in (1, 2):
        penalty = 0.15 if level == 2 else 0.0
        for row in range(1, 26):
            for col in range(1, 31):
                slots.append({
                    "slot_id": f"R{row:02d}C{col:02d}T{level}",
                    "row": row,
                    "col": col,
                    "level": level,
                    "manhattan_dist": round((row - 1) + (col - 1) + penalty, 2),
                })

    slots_df = pd.DataFrame(slots).sort_values("manhattan_dist").reset_index(drop=True)
    used_slots = set()
    assignments = []

    # Tezdeki üst kat kısıt eşikleri: ağırlık > 1500 veya desi > 1.00125 ise alt kat (Level 1)[cite: 20, 22]
    for _, item in df[df["abc_class"] == "A"].iterrows():
        force_level1 = (item["avg_weight_g"] > 1500.0) or (item["avg_desi"] > 1.00125)
        available = slots_df[~slots_df["slot_id"].isin(used_slots)]
        if force_level1:
            available = available[available["level"] == 1]
        if available.empty:
            continue

        chosen = available.iloc[0]
        used_slots.add(chosen["slot_id"])
        assignments.append({
            "material": item["material"],
            "ahp_score": round(float(item["ahp_score"]), 4),
            "force_level1": bool(force_level1),
            "slot_id": chosen["slot_id"],
            "level": int(chosen["level"]),
            "manhattan_dist": float(chosen["manhattan_dist"]),
        })

    return df, pd.DataFrame(assignments)


if __name__ == "__main__":
    orders = generate_synthetic_orders()
    scored_skus, rack_plan = run_analysis(orders)

    print("=== İLK 5 ÜRÜN (AHP SKORUNA GÖRE) ===")
    print(scored_skus[["material", "order_freq", "joint_order_ratio", "abc_class", "ahp_score"]].head(5).to_string(index=False))

    print("\n=== İLK 5 RAF ATAMASI ===")
    print(rack_plan.head(5).to_string(index=False))

    tot_imp = (THESIS_RESULTS["alt1_toplam_mesafe"] - THESIS_RESULTS["alt2_toplam_mesafe"]) / THESIS_RESULTS["alt1_toplam_mesafe"] * 100
    co_imp = (THESIS_RESULTS["alt1_birliktelik_mesafe"] - THESIS_RESULTS["alt2_birliktelik_mesafe"]) / THESIS_RESULTS["alt1_birliktelik_mesafe"] * 100

    print("\n=== TEZ DOĞRULAMA ÖZETİ ===")
    print(f"Toplam Temsili Mesafe: {THESIS_RESULTS['alt1_toplam_mesafe']:,.2f} -> {THESIS_RESULTS['alt2_toplam_mesafe']:,.2f} (%{tot_imp:.2f} iyileşme)")
    print(f"Birliktelik Mesafesi:  {THESIS_RESULTS['alt1_birliktelik_mesafe']:,.2f} -> {THESIS_RESULTS['alt2_birliktelik_mesafe']:,.2f} (%{co_imp:.2f} iyileşme)")