import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dir, save_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--min_user_interactions', type=int, default=20)
    parser.add_argument('--max_seq_len', type=int, default=50)
    return parser.parse_args()


def load_dat_files(data_dir: Path):
    ratings = pd.read_csv(
        data_dir / 'ratings.dat',
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1',
    )
    movies = pd.read_csv(
        data_dir / 'movies.dat',
        sep='::',
        engine='python',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1',
    )
    users = pd.read_csv(
        data_dir / 'users.dat',
        sep='::',
        engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        encoding='latin-1',
    )
    return ratings, movies, users


def extract_year(title: str) -> int:
    if isinstance(title, str) and title.endswith(')') and '(' in title:
        try:
            return int(title[-5:-1])
        except Exception:
            return 0
    return 0


def build_item_features(movies: pd.DataFrame, ratings: pd.DataFrame):
    movies = movies.copy()
    movies['year'] = movies['title'].map(extract_year)
    all_genres = sorted({g for gs in movies['genres'].fillna('') for g in gs.split('|') if g})
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}

    item_pop = ratings.groupby('movie_id').size().rename('popularity').reset_index()
    movie_mean = ratings.groupby('movie_id')['rating'].mean().rename('mean_rating').reset_index()
    movies = movies.merge(item_pop, on='movie_id', how='left').merge(movie_mean, on='movie_id', how='left')
    movies['popularity'] = movies['popularity'].fillna(0)
    movies['mean_rating'] = movies['mean_rating'].fillna(ratings['rating'].mean())
    movies['year'] = movies['year'].fillna(0)

    genre_mat = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    for row_idx, gs in enumerate(movies['genres'].fillna('').tolist()):
        for g in gs.split('|'):
            if g in genre_to_idx:
                genre_mat[row_idx, genre_to_idx[g]] = 1.0

    year_arr = movies['year'].to_numpy(dtype=np.float32)
    if year_arr.max() > year_arr.min():
        year_arr = (year_arr - year_arr.min()) / (year_arr.max() - year_arr.min())
    else:
        year_arr[:] = 0.0

    pop_arr = np.log1p(movies['popularity'].to_numpy(dtype=np.float32))
    pop_arr = pop_arr / (pop_arr.max() + 1e-8)

    mean_arr = movies['mean_rating'].to_numpy(dtype=np.float32) / 5.0

    side_features = np.concatenate(
        [genre_mat, year_arr[:, None], pop_arr[:, None], mean_arr[:, None]], axis=1
    )

    return movies, side_features, all_genres


def remap_ids(ratings, movies, users):
    user_ids = sorted(ratings['user_id'].unique().tolist())
    item_ids = sorted(ratings['movie_id'].unique().tolist())
    user2idx = {u: i + 1 for i, u in enumerate(user_ids)}
    item2idx = {m: i + 1 for i, m in enumerate(item_ids)}

    ratings = ratings.copy()
    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['item_idx'] = ratings['movie_id'].map(item2idx)

    movies = movies[movies['movie_id'].isin(item2idx)].copy()
    movies['item_idx'] = movies['movie_id'].map(item2idx)
    movies = movies.sort_values('item_idx').reset_index(drop=True)

    users = users[users['user_id'].isin(user2idx)].copy()
    users['user_idx'] = users['user_id'].map(user2idx)
    users = users.sort_values('user_idx').reset_index(drop=True)
    return ratings, movies, users, user2idx, item2idx


def build_sequences(ratings: pd.DataFrame, min_user_interactions: int):
    ratings = ratings.sort_values(['user_idx', 'timestamp', 'item_idx']).reset_index(drop=True)
    user_sequences = {}
    user_ratings = {}
    user_timestamps = {}

    for user_idx, grp in ratings.groupby('user_idx'):
        if len(grp) < min_user_interactions:
            continue
        user_sequences[int(user_idx)] = grp['item_idx'].astype(int).tolist()
        user_ratings[int(user_idx)] = grp['rating'].astype(float).tolist()
        user_timestamps[int(user_idx)] = grp['timestamp'].astype(int).tolist()

    valid_users = sorted(user_sequences.keys())
    return user_sequences, user_ratings, user_timestamps, valid_users


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    ratings, movies, users = load_dat_files(data_dir)
    movies, side_features_raw, all_genres = build_item_features(movies, ratings)
    ratings, movies, users, user2idx, item2idx = remap_ids(ratings, movies, users)

    # align side features to remapped item order
    movies_sorted = movies.sort_values('item_idx').reset_index(drop=True)
    old_movie_order = movies_sorted['movie_id'].tolist()
    raw_lookup = {mid: i for i, mid in enumerate(movies['movie_id'].tolist())}
    feat_dim = side_features_raw.shape[1]
    item_side_features = np.zeros((len(item2idx) + 1, feat_dim), dtype=np.float32)
    for _, row in movies_sorted.iterrows():
        item_side_features[int(row['item_idx'])] = side_features_raw[raw_lookup[int(row['movie_id'])]]

    user_sequences, user_ratings, user_timestamps, valid_users = build_sequences(
        ratings, args.min_user_interactions
    )

    item_counts = ratings.groupby('item_idx').size().reindex(range(1, len(item2idx) + 1), fill_value=0)
    propensity = item_counts.to_numpy(dtype=np.float32)
    propensity = propensity / propensity.sum()
    propensity = np.concatenate([[0.0], propensity], axis=0)

    payload = {
        'user_sequences': user_sequences,
        'user_ratings': user_ratings,
        'user_timestamps': user_timestamps,
        'valid_users': valid_users,
        'num_users': len(user2idx),
        'num_items': len(item2idx),
        'user2idx': user2idx,
        'item2idx': item2idx,
        'movies': movies_sorted,
        'users': users,
        'item_side_features': item_side_features,
        'item_propensity': propensity,
        'genre_vocab': all_genres,
        'max_seq_len': args.max_seq_len,
    }

    save_pickle(payload, out_dir / 'movielens_1m_preprocessed.pkl')
    print(f'Saved preprocessed data to: {out_dir / "movielens_1m_preprocessed.pkl"}')
    print(f'Users kept: {len(valid_users)}')
    print(f'Items: {len(item2idx)}')
    print(f'Item side feature dim: {item_side_features.shape[1]}')


if __name__ == '__main__':
    main()
