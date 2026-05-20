# Raw data drop-zone

Place the raw files here before running `python preprocess.py --phase 1`.

```
data/raw/
├── Beauty_5.json            # Amazon Beauty 5-core reviews
├── meta_Beauty.json         # Amazon Beauty item metadata
├── reviews_Beauty.json      # Amazon Beauty full reviews (optional)
└── ml-10M100K/
    ├── ratings.dat
    ├── movies.dat
    └── tags.dat
```

These files are not tracked in version control because of size.
Source URLs and SHA-256 checksums will be added to the top-level README
once the project is publicly released.
