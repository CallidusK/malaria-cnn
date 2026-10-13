# Learning notes

Plain-English notes on every concept used in this project, updated every phase.
Each note points to the file (and function) where the idea lives in the code.

---

## Phase 0: Project setup

### Virtual environment
**What:** a private folder (`.venv/`) holding this project's own copy of Python libraries.
**Analogy:** a separate toolbox for each job. Your LiDAR project can use one version of a library and this project another, without either breaking the other.
**How:** `python3 -m venv .venv` creates it; `source .venv/bin/activate` switches your terminal to use it. It's listed in `.gitignore` because everyone builds their own from `requirements.txt`.

### Random seeds
**What:** computers make "random" numbers with a formula that starts from a number called the seed. Same seed → same sequence of "random" numbers.
**Why we care:** ML is full of randomness (which patients go in the test set, the starting weights, which images get flipped). Fixing the seed means running the code twice gives the same results, so if a number changes, it's because *we* changed something.
**Where:** `src/utils.py` → `set_seed()`. Python, NumPy and PyTorch each have their own generator, so all three are seeded. The seed value is `RANDOM_SEED` in `src/config.py`.

### Device (CPU vs GPU)
**What:** the hardware that does the maths.
**Analogy:** a CPU is a few very smart workers; a GPU is thousands of simpler workers. Training a CNN is millions of simple multiplications, which is exactly the GPU's kind of job, so it's often 10 to 50 times faster.
**Where:** `src/utils.py` → `get_device()`. It tries NVIDIA GPU (`cuda`, what Kaggle has) → Apple GPU (`mps`) → plain `cpu`.

### Why everything lives in `config.py`
**What:** every setting (image size, epochs, learning rate, paths) is defined once in `src/config.py`.
**Why:** if the image size were typed separately in `data.py` and `models.py`, changing one and forgetting the other would cause a confusing crash. One source of truth means one place to change.
**Where:** `src/config.py`. It also detects Kaggle (`IS_ON_KAGGLE`) and picks the output folder automatically.

### Smoke test and fake data
**What:** a "smoke test" is a tiny, fast run that only checks the code doesn't crash (the name comes from electronics: power it on and see if it smokes).
**Why:** the real dataset only lives on Kaggle. Fake cartoon cells let us test the whole pipeline on a laptop in seconds.
**Where:** `src/make_fake_data.py` → `main()`. It deliberately copies the real dataset's quirks (nested duplicate folder, `Thumbs.db` junk files, patient codes in filenames) so our data code gets tested against them.
