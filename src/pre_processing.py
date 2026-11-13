# src/data.py
import os
import pandas as pd

# ========== CẤU HÌNH ==========
DATA_DIR = "data/raw/ml-1m"
OUT_DIR  = "data/processed"
RATING_THRESHOLD = 0        # 0 = mọi rating là tương tác; 4 = chỉ rating >= 4
K_CORE = 5                  # giữ user/item có >= 5 tương tác
MIN_RATINGS_PER_USER = 3    # cần >= 3 để có train + val + test khi leave-one-out

os.makedirs(OUT_DIR, exist_ok=True)


# ========== 1. ĐỌC DỮ LIỆU ==========
def load_ratings(path):
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


# ========== 2. CHUYỂN RATING -> TƯƠNG TÁC ==========
def to_implicit(df, threshold=RATING_THRESHOLD):
    """Giữ các cặp (user, item) thỏa ngưỡng; bỏ cột rating."""
    if threshold > 0:
        df = df[df["rating"] >= threshold]
    return df[["user_id", "movie_id", "timestamp"]].copy()


# ========== 3. LỌC K-CORE (LÀM TRƯỚC KHI CHIA!) ==========
def k_core_filter(df, k=K_CORE):
    """
    Lặp: giữ user có >= k tương tác và item có >= k tương tác
    cho đến khi không còn ai bị loại.
    """
    while True:
        n_before = len(df)
        u_counts = df["user_id"].value_counts()
        i_counts = df["movie_id"].value_counts()
        keep_users = u_counts[u_counts >= k].index
        keep_items = i_counts[i_counts >= k].index
        df = df[df["user_id"].isin(keep_users) & df["movie_id"].isin(keep_items)]
        if len(df) == n_before:  # ổn định
            break
    return df.reset_index(drop=True)


# ========== 4. CHIA LEAVE-ONE-OUT THEO THỜI GIAN ==========
def leave_one_out_split(df, min_per_user=MIN_RATINGS_PER_USER):
    """
    Với mỗi user, sắp theo timestamp:
      - tương tác cuối  -> test
      - tương tác kế cuối -> val
      - còn lại -> train
    User có < min_per_user tương tác bị loại (đã lọc k-core nên hiếm).
    """
    df = df.sort_values(["user_id", "timestamp"]).copy()
    df["rank_from_end"] = df.groupby("user_id").cumcount(ascending=False)

    # rank_from_end = 0 -> cuối (test), 1 -> val, >=2 -> train
    test  = df[df["rank_from_end"] == 0]
    val   = df[df["rank_from_end"] == 1]
    train = df[df["rank_from_end"] >= 2]

    # chỉ giữ user đủ 3 tập
    valid_users = set(train["user_id"]) & set(val["user_id"]) & set(test["user_id"])
    train = train[train["user_id"].isin(valid_users)]
    val   = val[val["user_id"].isin(valid_users)]
    test  = test[test["user_id"].isin(valid_users)]

    return (
        train[["user_id", "movie_id"]].reset_index(drop=True),
        val[["user_id", "movie_id"]].reset_index(drop=True),
        test[["user_id", "movie_id"]].reset_index(drop=True),
    )


# ========== 5. ĐÁNH LẠI ID TỪ 0 ==========
def remap_ids(train, val, test):
    """
    Xây mapping từ tập TRAIN (tránh item chỉ có ở test làm lệch index).
    Item/user chỉ xuất hiện ở val/test mà không có trong train sẽ bị loại
    (hoặc giữ nếu bạn muốn test cold-start — khi đó phải ghi rõ trong báo cáo).
    """
    users = sorted(train["user_id"].unique())
    items = sorted(train["movie_id"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    i2i = {m: i for i, m in enumerate(items)}

    def apply(df):
        df = df[df["user_id"].isin(u2i) & df["movie_id"].isin(i2i)].copy()
        df["user_id"] = df["user_id"].map(u2i)
        df["movie_id"] = df["movie_id"].map(i2i)
        return df.reset_index(drop=True)

    return apply(train), apply(val), apply(test), u2i, i2i


# ========== 6. LƯU FILE ==========
def save(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, sep="\t", header=False, index=False)
    print(f"  -> {path}  ({len(df):,} dòng)")


# ========== PIPELINE ==========
def main():
    print("[1] Đọc ratings.dat ...")
    ratings = load_ratings(os.path.join(DATA_DIR, "ratings.dat"))
    print(f"    gốc: {len(ratings):,} rating | "
          f"{ratings.user_id.nunique()} user | {ratings.movie_id.nunique()} item")

    print(f"[2] Chuyển sang implicit (ngưỡng rating >= {RATING_THRESHOLD}) ...")
    inter = to_implicit(ratings, RATING_THRESHOLD)
    print(f"    còn: {len(inter):,} tương tác")

    print(f"[3] Lọc {K_CORE}-core (TRƯỚC khi chia) ...")
    inter = k_core_filter(inter, K_CORE)
    print(f"    còn: {len(inter):,} tương tác | "
          f"{inter.user_id.nunique()} user | {inter.movie_id.nunique()} item")

    print("[4] Chia leave-one-out theo thời gian ...")
    train, val, test = leave_one_out_split(inter)
    print(f"    train={len(train):,} | val={len(val):,} | test={len(test):,}")

    print("[5] Đánh lại ID liên tục từ 0 ...")
    train, val, test, u2i, i2i = remap_ids(train, val, test)
    print(f"    user: {len(u2i)} | item: {len(i2i)}")
    print(f"    sau remap: train={len(train):,} | val={len(val):,} | test={len(test):,}")

    print("[6] Lưu ra data/processed/ ...")
    save(train, "train.txt")
    save(val,   "val.txt")
    save(test,  "test.txt")

    # lưu mapping để map ngược ra MovieID gốc khi cần
    pd.Series(u2i).rename("new_id").to_csv(
        os.path.join(OUT_DIR, "user_map.csv"), header=True)
    pd.Series(i2i).rename("new_id").to_csv(
        os.path.join(OUT_DIR, "item_map.csv"), header=True)

    print("Xong.")


if __name__ == "__main__":
    main()