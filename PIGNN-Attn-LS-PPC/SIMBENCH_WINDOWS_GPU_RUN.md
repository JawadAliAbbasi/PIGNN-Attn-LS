# SimBench Power Flow — Run the Model Yourself

This walks you through running a trained AC power-flow model on the SimBench
grid (94 buses), checking how accurate it is, and drawing comparison plots.

**You do not need a GPU.** Everything here runs on a normal laptop CPU in about
15 minutes. It works on Windows (PowerShell), on WSL Ubuntu, and on Linux/macOS.

**You do not need coding experience.** Every step is one command to copy and
paste. After each step there is a line telling you what you should see.

At the end you will have plots like this — one per scenario, comparing the
model against the reference solver bus by bus:

```
            |V|            theta            P              Q
PIGNN     [ graph ]      [ graph ]      [ graph ]      [ graph ]
DC init   [ graph ]      [ graph ]      [ graph ]      [ graph ]
Newton    [ graph ]      [ graph ]      [ graph ]      [ graph ]
```

---

## Before you start

1. **Turn on the FAU VPN.** Nothing below works without it.
2. Make sure you have about **1.5 GB of free disk space**.
3. Open a terminal:
   - Windows: press Start, type `PowerShell`, open it
   - WSL / Linux / macOS: open your normal terminal

You will need the password for `chkim@131.188.35.62`. Ask the project owner.

---

## Step 1 — Download everything (one command)

Everything you need — data, model, scripts — comes down in a single command.

**WSL / Linux / macOS:**

```bash
scp -r chkim@131.188.35.62:PIGNN-Attn-LS/simbench_v2_release ~/simbench
```

**Windows PowerShell:**

```powershell
scp -r chkim@131.188.35.62:PIGNN-Attn-LS/simbench_v2_release $HOME\simbench
```

Type the password when asked. This downloads about 450 MB, usually 2–10 minutes.

Now go into the folder:

```bash
cd ~/simbench
```

(Windows PowerShell: `cd $HOME\simbench`)

**What you should see** — run `ls` (PowerShell: `dir`):

```
CONFIG.txt   README.md   ckpt   data   environment.yml   requirements.txt   scripts
```

### Check the data file arrived complete

A half-finished download does not report an error, so check the size:

```bash
ls -l data/
```

The number must be exactly **448628631**. If it is anything else, delete the
folder and run the `scp` command again.

---

## Step 2 — Set up Python

Pick **one** of the two options.

### Option A — conda (recommended if you have Anaconda or Miniconda)

```bash
conda env create -f environment.yml
conda activate simbench-pf
```

Then install PyTorch:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

### Option B — plain Python (no conda)

```bash
python -m venv simbench-env
```

Activate it:

```bash
source simbench-env/bin/activate        # WSL / Linux / macOS
.\simbench-env\Scripts\Activate.ps1     # Windows PowerShell
```

Then:

```bash
pip install -r requirements.txt
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

### Check it worked

```bash
python -c "import torch, torch_scatter, pyarrow, matplotlib; print('all good, torch', torch.__version__)"
```

**What you should see:** `all good, torch 2.4.0+cpu`

> **Have an NVIDIA GPU and want to use it?** Replace `cpu` with `cu121` in both
> `pip install` lines above. It is faster but not required — and the CPU version
> is much less likely to give you installation trouble.

---

## Step 3 — Quick test (2 scenarios, about 1 minute)

Before the long run, check everything is wired up:

```bash
python scripts/plot_bus_profiles.py \
  --PARQUET data/SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_38649_NR_branchrows_directSI.parquet \
  --ckpt_pignn ckpt/pfv2_pignn_SimBench_b32_40_best_model.ckpt \
  --n 100 --limit 2 \
  --out_dir results
```

On Windows PowerShell, replace each `\` at the end of a line with a backtick `` ` ``.

**What you should see**, among other output:

```
[split] test=12887
[models] ['PIGNN']
done: 2 scenarios -> results
```

Now open the picture that was created:

```
results/scenario_000_row000035/bus_profiles.png
```

It should show 12 graphs in 3 rows and 4 columns. The blue line is the
reference answer; the red dashed line is what the model produced. In the top
row (PIGNN) the two lines should sit almost exactly on top of each other.

If you got here, everything works.

---

## Step 4 — Measure the accuracy

This runs the model over the whole held-out test set and prints the numbers:

```bash
python scripts/rescore_pignn.py \
  --PARQUET data/SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_38649_NR_branchrows_directSI.parquet \
  --ckpt ckpt/pfv2_pignn_SimBench_b32_40_best_model.ckpt \
  --grid SimBench \
  --BATCH 8 \
  --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
  --armijo_mode geometric_safe \
  --physics_loss_form logcosh \
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value 42 \
  --max_test_samples 400
```

**What you should see:** one long line beginning with `RESCORE`, followed by
results in `{"name": number, ...}` form. Four of them matter:

| Name in the output | What it means | Expected |
|---|---|---|
| `rmse_vmag_pu` | voltage magnitude error | about `1.5e-03` |
| `rmse_theta_deg` | voltage angle error | about `1.0e-01` |
| `R2_vmag` | how much of the voltage variation the model explains, 1.0 is perfect | about `0.999` |
| `slope_vmag` | 1.0 means it captures the full swing, 0 means it just predicts an average | about `0.997` |

The last two are the real test. A model can post a small error simply by
guessing "every bus is at 1.0" — `R2_vmag` and `slope_vmag` near 1.0 are what
show it is genuinely following the answer.

`--max_test_samples 400` keeps this to a couple of minutes. Remove that line to
score all 12,887 test scenarios instead; that takes 30–60 minutes on a CPU and
gives `rmse_vmag_pu` 1.52e-03 and `rmse_theta_deg` 9.96e-02, the published
figures.

Power mismatch numbers are not printed here — they come from Step 5.

> If the numbers are far off, the most likely cause is a changed setting.
> `--armijo_mode geometric_safe` in particular must stay as written: a
> different value still loads the model but changes what it computes.

---

## Step 5 — Make all 100 plots

Same command as Step 3, just without `--limit 2`:

```bash
python scripts/plot_bus_profiles.py \
  --PARQUET data/SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_38649_NR_branchrows_directSI.parquet \
  --ckpt_pignn ckpt/pfv2_pignn_SimBench_b32_40_best_model.ckpt \
  --n 100 \
  --out_dir results
```

This takes about 10–15 minutes on a CPU. It creates one folder per scenario:

```
results/scenario_000_row000035/bus_profiles.png     the picture
results/scenario_000_row000035/metrics.json         that scenario's numbers
results/summary.json                                all 100 scenarios
```

Roughly 100 MB in total.

---

## How to read the plots

Each picture is one scenario — one snapshot of the grid. The horizontal axis is
the bus number, 0 to 93.

**The columns** are the four quantities:

| Column | Meaning |
|---|---|
| `|V|` | voltage magnitude at each bus |
| `theta` | voltage angle at each bus |
| `P` | active power injected at each bus |
| `Q` | reactive power injected at each bus |

**The rows** are three different ways of getting an answer:

| Row | What it is |
|---|---|
| `PIGNN` | the trained model — this is what we are testing |
| `DC init` | the rough starting guess, before any solving. The "do nothing" baseline |
| `custom NR` | a classical Newton-Raphson solver, for comparison |

**Blue is always the reference answer** (pandapower, the standard solver).
Red dashed is the row's own answer. The `RMSE` in each title is the average
error in that panel — smaller is better.

What you should notice:

- **PIGNN** sits almost on top of the blue line.
- **DC init** is visibly flat and wrong — that is the point, it shows how much
  work the model is actually doing.
- **custom NR** matches blue to about 14 decimal places, which confirms the
  comparison itself is set up correctly.

Typical values across the 100 scenarios:

| Row | `|V|` error | `theta` error |
|---|---|---|
| PIGNN | 7.06e-04 pu | 0.044 degrees |
| DC init | 5.25e-02 pu | 2.55 degrees |
| custom NR | 2.15e-14 pu | 1.10e-12 degrees |

So the model is about **74 times more accurate than the starting guess** on
voltage magnitude, and about **58 times** on angle.

---

## How the model was trained

The full record is in `CONFIG.txt`. The short version:

| Item | Value |
|---|---|
| model | `GNSMsg_EdgeSelfAttn` (PIGNN-Attn-LS), 65,896 parameters |
| grid | SimBench, 94 buses |
| dataset | ppNR v2 corpus, 38,649 scenarios, pandapower NR labels |
| test split | 12,887 scenarios (seed 42) |
| epochs | 40 |
| batch / learning rate | 32 / 1e-4 |
| solver steps `K` | 40 |
| line search | Armijo, `geometric_safe` |
| loss | MSE + physics residual (logcosh) |

---

## If something goes wrong

**`scp` asks for a password over and over**
You are probably not on the FAU VPN. Connect and try again.

**The data file size is not 448628631**
The download was cut short. Delete the folder and redo Step 1.

**`ModuleNotFoundError: torch_scatter`**
The `torch-scatter` line in Step 2 did not run or did not match your PyTorch
version. Check with `python -c "import torch; print(torch.__version__)"` and
use that version number in the wheel URL, for example
`https://data.pyg.org/whl/torch-2.4.0+cpu.html`.

**`ModuleNotFoundError: gridfm_graphkit` or `gridsfm` or `lumina_inference`**
Those are only needed for three other models that are not part of this package.
Make sure your command has only `--ckpt_pignn` and no other `--ckpt_*` option.

**It is very slow**
Normal on CPU: about 8 seconds per scenario. Use `--limit 10` to do fewer.

**The plots look empty or the run crashes near the end**
Make sure `scripts/newton_raphson_improved.py` is still in the `scripts` folder.
It is needed for the bottom row.
