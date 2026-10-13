import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"c:\Users\ashis\OneDrive\Desktop\pp1\NIDS-Reliability")
HTML_PATH = ROOT / "Part_B_Implementation_Plan.html"
PDF_PATH = ROOT / "Part_B_Implementation_Plan.pdf"

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Part B Implementation Plan - NIDS Reliability Evaluation</title>
<style>
  @page {
    size: A4 portrait;
    margin: 12mm 13mm 12mm 13mm;
  }

  * {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.45;
    font-size: 8.2pt;
    margin: 0;
    padding: 0;
  }

  .header-card {
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-bottom: 11px;
  }
  .header-title {
    font-size: 15pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 2px 0;
    letter-spacing: -0.02em;
  }
  .header-subtitle {
    font-size: 9.3pt;
    font-weight: 600;
    color: #2563eb;
    margin: 0 0 6px 0;
  }
  .meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-top: 6px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 7.4pt;
  }
  .meta-item strong {
    color: #64748b;
    display: block;
    text-transform: uppercase;
    font-size: 6.4pt;
    letter-spacing: 0.05em;
    margin-bottom: 1px;
  }
  .meta-item span {
    color: #0f172a;
    font-weight: 600;
  }

  h2 {
    font-size: 10pt;
    font-weight: 700;
    color: #0f172a;
    border-left: 3.5px solid #2563eb;
    padding-left: 6px;
    margin-top: 11px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }
  h3 {
    font-size: 8.6pt;
    font-weight: 700;
    color: #1e293b;
    margin-top: 7px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }

  p {
    margin: 0 0 5px 0;
    text-align: justify;
  }

  ul, ol {
    margin: 0 0 5px 0;
    padding-left: 17px;
  }
  li {
    margin-bottom: 2px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 8px 0;
    font-size: 7.5pt;
    page-break-inside: avoid;
  }
  th {
    background-color: #0f172a;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 4px 6px;
    font-size: 7.1pt;
    letter-spacing: 0.01em;
  }
  td {
    padding: 3.6px 6px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
    vertical-align: top;
  }
  tr:nth-child(even) td {
    background-color: #f8fafc;
  }

  .callout {
    border-radius: 4px;
    padding: 6px 9px;
    margin: 6px 0;
    font-size: 7.5pt;
    page-break-inside: avoid;
  }
  .callout-info {
    background-color: #eff6ff;
    border-left: 3px solid #2563eb;
    color: #1e40af;
  }
  .callout-warning {
    background-color: #fffbeb;
    border-left: 3px solid #d97706;
    color: #92400e;
  }
  .callout-gate {
    background-color: #f0fdf4;
    border-left: 3px solid #16a34a;
    color: #166534;
  }
  .callout strong {
    font-weight: 700;
  }

  .equation-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 5px 8px;
    margin: 5px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 7.8pt;
    color: #0f172a;
    display: flex;
    justify-content: space-around;
    page-break-inside: avoid;
  }
  .eq-item {
    font-style: italic;
  }
  .eq-item span {
    font-style: normal;
  }

  code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 7.3pt;
    background-color: #f1f5f9;
    padding: 1px 3px;
    border-radius: 2px;
    color: #0f172a;
  }
  .code-block {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 6px 8px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 6.7pt;
    color: #0f172a;
    line-height: 1.28;
    margin: 5px 0;
    page-break-inside: avoid;
  }

  .badge {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 6.5pt;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-amber { background: #fef3c7; color: #b45309; }

  .page-break {
    page-break-before: always;
  }
  .section-block {
    page-break-inside: avoid;
    margin-bottom: 7px;
  }
</style>
</head>
<body>

<div class="header-card">
  <div class="header-title">Major Project Implementation Plan &mdash; Part B (Audited &amp; Refined)</div>
  <div class="header-subtitle">Reliability Evaluation of Machine Learning-Based NIDS Under Temporal and Controlled Stress Conditions</div>
  <div class="meta-grid">
    <div class="meta-item">
      <strong>Protocol Scope</strong>
      <span>UNSW-NB15 &amp; Cross-Dataset</span>
    </div>
    <div class="meta-item">
      <strong>Baseline Checkpoint</strong>
      <span>2fe4892 (CICIDS2017 Validated)</span>
    </div>
    <div class="meta-item">
      <strong>Models &amp; Seeds</strong>
      <span>LR, RF, XGBoost | 5 Seeds</span>
    </div>
    <div class="meta-item">
      <strong>Status</strong>
      <span><span class="badge badge-amber">Awaiting Approval &amp; Sync</span></span>
    </div>
  </div>
</div>

<div class="section-block">
  <h2>1. Executive Summary &amp; Research Mandate</h2>
  <p>
    This document presents the revised, empirical implementation plan for <strong>Part B</strong> of the research project <em>"Reliability Evaluation of Machine Learning-Based Network Intrusion Detection Systems Under Temporal and Controlled Stress Conditions"</em>.
    Following the successful completion and documentation of Part A (CICIDS2017) at checkpoint <code>2fe4892</code>, Part B integrates <strong>UNSW-NB15</strong> and executes a systematic <strong>cross-dataset reliability evaluation</strong>. Crucially, this refined plan incorporates an audit of the actual UNSW-NB15 files, individual justifications for feature exclusions (particularly ports), exact leakage-safe preprocessing specifications, transparent Logistic Regression convergence handling, and explicit recognition that the temporal conditions across the two datasets represent fundamentally different phenomena.
  </p>
  <div class="callout callout-info">
    <strong>Scientific Guardrails (AGENTS.md):</strong> Zero synthetic results. All learned transformations fitted strictly on training data. No flow duration or proxy feature substituted as a timestamp. Binary and multiclass tasks remain strictly partitioned. No claims of concept drift proof, universal model superiority, or cross-dataset transferability.
  </div>
</div>

<div class="section-block">
  <h2>2. Empirical Audit of Actual UNSW-NB15 Files &amp; Provenance</h2>
  <p>
    An empirical inspection of candidate distributions on the Hugging Face Hub (including <code>Mouwiya/UNSW-NB15-small</code>, <code>lacg030175/UNSW-NB15</code>, and official ACCS releases) reveals key structural realities that govern the experiment:
  </p>

  <table>
    <thead>
      <tr>
        <th style="width: 26%;">Distribution Format</th>
        <th style="width: 25%;">File Names &amp; Sizes</th>
        <th style="width: 22%;">Row Count &amp; Features</th>
        <th style="width: 27%;">Timestamp &amp; Duplicate Findings</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Standard Partitioned Set</strong> (Official ACCS Benchmark)</td>
        <td><code>UNSW_NB15_training-set.csv</code> (32.3 MB)<br><code>UNSW_NB15_testing-set.csv</code> (15.4 MB)</td>
        <td>175,341 rows (Train)<br>82,332 rows (Test)<br>Total: 257,673 rows<br>45 columns (with <code>id</code>)</td>
        <td><strong>Timestamps absent:</strong> <code>Stime</code> and <code>Ltime</code> were stripped by ACCS.<br><strong>Severe duplication:</strong> 67,601 duplicates (38.56%) in training, 26,387 (32.05%) in testing without <code>id</code>. 48,111 cross-partition duplicates.</td>
      </tr>
      <tr>
        <td><strong>Full Raw Flow Set</strong> (4 CSVs / Full Parquet)</td>
        <td><code>UNSW-NB15_1.csv</code> to <code>4.csv</code><br>Total ~500 MB uncompressed</td>
        <td>2,540,044 raw records<br>1,584,259 deduplicated<br>49 columns</td>
        <td><strong>Timestamps present:</strong> <code>Stime</code> and <code>Ltime</code> epoch seconds.<br><strong>Session provenance:</strong> Contains 2 discrete capture periods. 30.5% duplicate flow records removed during cleaning.</td>
      </tr>
    </tbody>
  </table>

  <h3>A. Timestamp Ranges &amp; Temporal-Split Feasibility Policy</h3>
  <ul>
    <li><strong>Data-Driven Provenance (No Blind Assumptions):</strong> We will not assume a January/February split until the selected dataset's timestamp distribution is verified at runtime.</li>
    <li><strong>Dual-Mode Provenance Detection:</strong>
      <ol>
        <li><em>If raw flow data with <code>Stime</code> is used:</em> Parse epoch values, verify monotonicity, plot/audit the timestamp histogram, and construct the temporal holdout at a verified session boundary (e.g., Session 1 train vs. Session 2 test).</li>
        <li><em>If the standard 45-column benchmark distribution is used:</em> Acknowledge that row timestamps are absent. Temporal evaluation in this mode utilizes the official ACCS train/test holdout, where train and test were captured during separate traffic sessions. Flow duration (<code>dur</code>) or packet IAT will <strong>never</strong> be used as a fake timestamp.</li>
      </ol>
    </li>
  </ul>
</div>

<div class="page-break"></div>

<div class="section-block">
  <h2>3. Individual Justification for Feature Exclusions</h2>
  <p>
    Every excluded feature in the pipeline is eliminated based on specific, documented scientific rationales rather than arbitrary omission:
  </p>

  <table>
    <thead>
      <tr>
        <th style="width: 18%;">Feature Name</th>
        <th style="width: 18%;">Type in Raw Data</th>
        <th style="width: 64%;">Exclusion Rationale &amp; Leakage Prevention</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>srcip</code>, <code>dstip</code></td>
        <td>Nominal (IP Strings)</td>
        <td><strong>Spatial Memorization Leakage:</strong> Encodes network topology and subnet identity. Classifiers split on specific IP strings, memorizing attacker/victim VMs in the synthetic testbed. Fails completely in cross-network or shifted environments.</td>
      </tr>
      <tr>
        <td><code>sport</code></td>
        <td>Integer (Source Port)</td>
        <td><strong>Ephemeral Port Noise:</strong> Assigned dynamically client-side by OS socket allocators (typically 32,768&ndash;65,535). High cardinality, ephemeral noise with zero intrinsic intrusion semantics; causes decision trees to overfit to client socket allocation states.</td>
      </tr>
      <tr>
        <td><code>dsport</code></td>
        <td>Integer (Destination Port)</td>
        <td><strong>Lab Target Bias &amp; Feature Redundancy:</strong> While destination ports represent targeted services (e.g., 80, 22), treating raw port numbers as continuous integers is statistically invalid (port 81 is not "greater than" 80). Furthermore, raw ports leak synthetic lab server configurations. <strong>Crucially, UNSW-NB15 already extracts 6 engineered features capturing port behaviors:</strong> <code>service</code> (nominal), <code>ct_srv_src</code>, <code>ct_srv_dst</code>, <code>ct_src_dport_ltm</code>, <code>ct_dst_sport_ltm</code>, and <code>is_sm_ips_ports</code>. Dropping raw <code>dsport</code> prevents lab overfitting while retaining full behavioral port context.</td>
      </tr>
      <tr>
        <td><code>id</code></td>
        <td>Integer (Index)</td>
        <td><strong>Monotonic Key Leakage:</strong> Synthetic row identifier added during export. Serves as a perfect shortcut predictor if retained.</td>
      </tr>
      <tr>
        <td><code>Stime</code>, <code>Ltime</code></td>
        <td>Timestamp / Epoch</td>
        <td><strong>Time-Memorization Leakage:</strong> Retained strictly for chronological sorting and temporal partitioning. Must be excluded from feature matrix $X$ so models learn traffic dynamics rather than epoch time thresholds.</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="section-block">
  <h2>4. Exact Leakage-Safe Numeric &amp; Categorical Preprocessing Pipeline</h2>
  <p>
    The preprocessing sequence strictly isolates the training partition from evaluation data across all six stress conditions:
  </p>

  <ol>
    <li><strong>Split Before Preprocessing:</strong> Train and test index partitions are established prior to any transformation. No global statistics are computed.</li>
    <li><strong>Categorical Preprocessing (Unseen Test-Category Handling):</strong>
      <ul>
        <li>Nominal features: <code>proto</code> (protocol), <code>service</code> (service), <code>state</code> (connection state).</li>
        <li>Transformed via <code>OneHotEncoder(handle_unknown="ignore", sparse_output=False)</code> fitted <strong>strictly on $X_{\text{train}}$</strong>.</li>
        <li><strong>Unseen Category Protocol:</strong> If an unseen protocol or state appears in test data (or under missingness corruption), <code>handle_unknown="ignore"</code> outputs an all-zero vector across the dummy columns for that feature. No test-category information leaks into the vocabulary.</li>
      </ul>
    </li>
    <li><strong>Numeric Imputation:</strong>
      <ul>
        <li>39 continuous and engineered count features are handled via <code>SimpleImputer(strategy="median", keep_empty_features=True)</code> fitted <strong>strictly on $X_{\text{train}}$</strong>.</li>
        <li>The fitted imputer transforms both clean test data and missingness-corrupted test matrices. Zero non-finite values (<code>NaN</code>, <code>Inf</code>) are permitted in estimator inputs.</li>
      </ul>
    </li>
    <li><strong>Model-Specific Scaling:</strong>
      <ul>
        <li><strong>Logistic Regression:</strong> <code>StandardScaler</code> fitted solely on imputed $X_{\text{train}}$ and applied to evaluation sets.</li>
        <li><strong>Random Forest &amp; XGBoost:</strong> Invariant to monotonic scaling; receive imputed float32 feature arrays directly.</li>
      </ul>
    </li>
    <li><strong>Test-Time Controlled Missingness Stress:</strong>
      <ul>
        <li>Independent Bernoulli masks ($p \in \{0.10, 0.20\}$) are applied <strong>exclusively to evaluation matrices</strong> at test time.</li>
        <li>Injected NaNs are passed through the pre-fitted train median imputer. Training data and fitted models are never modified.</li>
      </ul>
    </li>
  </ol>
</div>

<div class="page-break"></div>

<div class="section-block">
  <h2>5. Class Coverage, Sample Sizes &amp; Non-Equivalence of Temporal Conditions</h2>
  <p>
    A critical scientific contribution of this evaluation is acknowledging that <strong>CICIDS2017 and UNSW-NB15 temporal conditions represent fundamentally different phenomena</strong>:
  </p>

  <table>
    <thead>
      <tr>
        <th style="width: 20%;">Evaluation Dimension</th>
        <th style="width: 25%;">CICIDS2017 Baseline (Part A)</th>
        <th style="width: 25%;">UNSW-NB15 Evaluation (Part B)</th>
        <th style="width: 30%;">Scientific Implication &amp; Non-Equivalence</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Reference Split</strong></td>
        <td>1,671,939 Train / 417,985 Test<br>(80/20 Stratified across 27 classes)</td>
        <td>Standard: 175,341 Train / 82,332 Test<br>Deduplicated: ~1.26M Train / ~316K Test</td>
        <td>Both evaluate in-distribution performance under identical stratified sampling.</td>
      </tr>
      <tr>
        <td><strong>Temporal Split</strong></td>
        <td>1,543,723 Train (Mon&ndash;Thu)<br>546,201 Test (Friday only)</td>
        <td>Standard Holdout: 175,341 Train / 82,332 Test<br>Verified Session Cut: Session 1 / Session 2</td>
        <td><strong>Fundamental difference in class dynamics:</strong><br>&bull; <em>CICIDS2017:</em> 22 attack classes absent from Friday test; 4 attacks absent from Mon&ndash;Thu train. Macro-F1 bounded near 0.037.<br>&bull; <em>UNSW-NB15:</em> <strong>All 10 classes present in both partitions.</strong> Tests true distribution shift under continuous class coverage.</td>
      </tr>
      <tr>
        <td><strong>Class Coverage</strong></td>
        <td>5 of 27 classes evaluated on Friday</td>
        <td>10 of 10 classes evaluated in temporal split</td>
        <td>Temporal degradation cannot be compared naively; report must distinguish unseen-class ceilings from feature drift.</td>
      </tr>
    </tbody>
  </table>

  <h3>UNSW-NB15 Class Distribution (Audited Benchmark Split)</h3>
  <table>
    <thead>
      <tr>
        <th>Class Name</th>
        <th>Train Count (175k)</th>
        <th>Test Count (82k)</th>
        <th>Total Count</th>
        <th>Temporal Status</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Normal</strong></td><td>56,000</td><td>37,000</td><td>93,000</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Generic</strong></td><td>40,000</td><td>18,871</td><td>58,871</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Exploits</strong></td><td>33,393</td><td>11,132</td><td>44,525</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Fuzzers</strong></td><td>18,184</td><td>6,062</td><td>24,246</td><td>Present in both partitions</td></tr>
      <tr><td><strong>DoS</strong></td><td>12,264</td><td>4,089</td><td>16,353</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Reconnaissance</strong></td><td>10,491</td><td>3,496</td><td>13,987</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Analysis</strong></td><td>2,000</td><td>677</td><td>2,677</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Backdoor</strong></td><td>1,746</td><td>583</td><td>2,329</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Shellcode</strong></td><td>1,133</td><td>378</td><td>1,511</td><td>Present in both partitions</td></tr>
      <tr><td><strong>Worms</strong></td><td>130</td><td>44</td><td>174</td><td>Present in both partitions</td></tr>
    </tbody>
  </table>
</div>

<div class="section-block">
  <h2>6. Mathematical Definitions &amp; Handling of Edge Cases</h2>
  <ul>
    <li><strong>Undefined Metrics &amp; Absent Classes:</strong>
      <ul>
        <li><code>precision_recall_fscore_support(..., zero_division=0)</code>: If a class has zero true instances in the test partition (support = 0), recall is defined as 0.0 under <code>zero_division=0</code>. If never predicted, precision is 0.0.</li>
        <li>Macro metrics represent unweighted means over all $K$ target classes: $\text{Macro-F1} = \frac{1}{K}\sum_{i=1}^K F1_i$.</li>
      </ul>
    </li>
    <li><strong>Zero Denominators in Degradation:</strong>
      <ul>
        <li>Formula: $D = (M_{\text{Reference}} - M_{\text{Condition}}) / M_{\text{Reference}}$.</li>
        <li>Handled via <code>.replace(0, np.nan)</code>: if $M_{\text{Reference}} = 0$, degradation is mathematically undefined and recorded as <code>NaN</code> (never fabricated as 0.0 or 1.0).</li>
      </ul>
    </li>
    <li><strong>False Positive Rate (FPR):</strong>
      <ul>
        <li>Per-class $FPR_i = FP_i / (FP_i + TN_i)$. If $FP_i + TN_i = 0$, $FPR_i = 0.0$. Macro-FPR is unweighted mean across all classes.</li>
        <li>Relative FPR Degradation: $D_{\text{FPR}} = (M_{\text{Condition}} - M_{\text{Reference}}) / M_{\text{Reference}}$ with zero denominators replaced by <code>NaN</code>.</li>
      </ul>
    </li>
    <li><strong>Non-Additive Interaction Term:</strong>
      $$\text{Interaction} = D_{\text{Combined}} - (D_{\text{Temporal}} + D_{\text{Missingness}})$$
    </li>
    <li><strong>Student-t 95% Confidence Intervals:</strong>
      $$\bar{x} \pm t_{0.975, df=4} \cdot \frac{s}{\sqrt{5}} \quad (t_{\text{crit}} = 2.7764)$$
      Sample standard deviation $s$ uses Bessel correction ($ddof=1$). If $s = 0$, CI half-width is 0.0. If $N < 2$, CI bounds are <code>NaN</code>.
    </li>
  </ul>
</div>

<div class="page-break"></div>

<div class="section-block">
  <h2>7. Logistic Regression Convergence Transparency Protocol</h2>
  <p>
    Rather than claiming that iteration counts prove mathematical convergence, we establish an explicit convergence audit:
  </p>
  <ol>
    <li><strong>Active Warning Interception:</strong> `model.fit()` runs within `warnings.catch_warnings(record=True)` to capture scikit-learn `ConvergenceWarning` events.</li>
    <li><strong>Diagnostic Recording:</strong> The export metadata records:
      <ul>
        <li><code>lr_iterations</code>: Actual iterations executed (<code>int(model.n_iter_[0])</code>).</li>
        <li><code>lr_max_iter</code>: Iteration ceiling (<code>300</code>).</li>
        <li><code>lr_converged_without_warning</code>: Boolean indicating whether L-BFGS terminated within tolerance before reaching <code>max_iter</code>.</li>
        <li><code>lr_warnings_logged</code>: Verbatim warning strings if any were raised.</li>
      </ul>
    </li>
    <li><strong>Scientific Disclosure:</strong> The report explicitly states that terminating below <code>max_iter</code> indicates the gradient norm satisfied stopping tolerance `tol=1e-3`, while reaching <code>max_iter=300</code> indicates an iteration cutoff. We explicitly avoid claiming that `n_iter_` alone constitutes mathematical proof of global optimality.</li>
  </ol>
</div>

<div class="section-block">
  <h2>8. Cross-Dataset Comparative Analysis Framework</h2>
  <p>
    The cross-dataset synthesis answers four core research questions:
  </p>
  <ol>
    <li><strong>Structural Ceiling vs. Feature Drift:</strong> Disentangles whether temporal vulnerability is driven by unseen classes (CICIDS2017) or continuous feature drift across identical classes (UNSW-NB15).</li>
    <li><strong>Architectural Resilience:</strong> Evaluates whether gradient-boosted trees (XGBoost) and random forests maintain consistent superiority over linear models across both network topologies.</li>
    <li><strong>Interaction Geometry:</strong> Assesses whether combined stress exhibits sub-additive saturation (floor effects) or super-additive degradation across datasets.</li>
    <li><strong>Scientific Boundaries (Rule 8):</strong> Clarifies that findings highlight dataset-specific reliability profiles rather than universal model superiority or production readiness.</li>
  </ol>
</div>

<div class="section-block">
  <h2>9. Codebase Architecture &amp; File Modifications</h2>
  <div class="code-block">
NIDS-Reliability/
├── configs/
│   ├── experiment_config.yaml         # Preserved (CICIDS2017 reference configuration)
│   └── unsw_experiment_config.yaml    # Standalone UNSW-NB15 experimental parameters
├── src/
│   ├── 01_setup.py ... 07_*.py        # CICIDS2017 pipeline (Preserved intact)
│   ├── unsw_01_setup.py               # UNSW-NB15 automatic downloader &amp; schema auditor
│   ├── unsw_02_data_and_temporal.py   # Audited ingestion, stime sorting &amp; duplicate cleaning
│   ├── unsw_03_preprocessing.py       # Imputers, encoders, scalers &amp; model factories
│   ├── unsw_04_experiments.py         # 5-seed 6-condition UNSW-NB15 runner
│   ├── unsw_05_analysis.py            # Degradation, interaction &amp; attack-level metrics
│   ├── unsw_06_repeated_figures.py    # Statistical aggregation (95% CI) &amp; Figures 1-6
│   ├── unsw_07_validation_export.py   # Integrity assertions &amp; structured CSV/JSON exports
│   └── cross_dataset_analysis.py      # Cross-dataset comparative synthesis &amp; visualizations
├── run_experiments.py                 # Existing CICIDS2017 execution script
├── run_unsw_experiments.py            # Dedicated execution script for UNSW-NB15
├── run_cross_analysis.py              # Execution script for cross-dataset analysis
├── diagnostic_audit.py                # Updated to audit both datasets &amp; fix cp1252 output
├── UNSW_EXPERIMENT_DESIGN.md          # Full methodology documentation for UNSW-NB15
└── CROSS_DATASET_REPORT.md            # Comprehensive comparative report for final thesis
  </div>
</div>

<div class="page-break"></div>

<div class="section-block">
  <h2>10. Complete Inventory of Deliverables &amp; Generated Artifacts</h2>
  <table>
    <thead>
      <tr>
        <th style="width: 38%;">Output File / Directory</th>
        <th style="width: 24%;">Dimensions / Content</th>
        <th style="width: 38%;">Description &amp; Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>results/unsw_nb15/all_results_seed42.csv</code></td>
        <td>36 rows (2 &times; 3 &times; 6)</td>
        <td>Per-combination metrics for primary seed 42</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/repeated_seed_results.csv</code></td>
        <td>180 rows</td>
        <td>Authoritative raw data across all 5 seeds</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/repeated_seed_summary.csv</code></td>
        <td>36 rows &times; 15 cols</td>
        <td>5-seed means, standard deviations, and 95% CIs</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/final_condition_metrics.csv</code></td>
        <td>36 rows</td>
        <td>Aggregated performance baseline per condition</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/degradation_*.csv</code></td>
        <td>36 rows per metric</td>
        <td>Relative degradation tables (F1, Recall, FPR)</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/interaction_results.csv</code></td>
        <td>12 rows (2 &times; 3 &times; 2)</td>
        <td>Non-additive combined stress interaction metrics</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/attack_level_recall.csv</code></td>
        <td>180 rows (10 &times; 6 &times; 3)</td>
        <td>Per-attack-class detection recall under all conditions</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/figure1-6_*.png</code></td>
        <td>6 high-res figures</td>
        <td>Publication-quality charts matching Part A specs</td>
      </tr>
      <tr>
        <td><code>results/unsw_nb15/validation.json</code></td>
        <td>JSON schema</td>
        <td>Audit metadata confirming zero-leakage, pass flags, and LR convergence log</td>
      </tr>
      <tr>
        <td><code>results/cross_dataset/comparison_*.csv</code></td>
        <td>Comparative tables</td>
        <td>Side-by-side performance, degradation &amp; interactions</td>
      </tr>
      <tr>
        <td><code>results/cross_dataset/figure_cross_*.png</code></td>
        <td>Comparative figures</td>
        <td>Visual comparison of stress vulnerabilities across datasets</td>
      </tr>
      <tr>
        <td><code>CROSS_DATASET_REPORT.md</code></td>
        <td>Full Academic Text</td>
        <td>Comprehensive final-year report chapter with limitations</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="section-block">
  <h2>11. Scientific Validation &amp; Leakage Verification Criteria</h2>
  <ol>
    <li><strong>Zero Data Leakage:</strong> Encoders, imputers, and scalers fitted strictly on training subsets. Temporal partitions isolated prior to any preprocessing.</li>
    <li><strong>Unseen Category Defense:</strong> All-zero dummy encoding for test categories unseen during training; verified zero encoding exceptions.</li>
    <li><strong>Matrix Cleanliness:</strong> Automated assertions confirming zero <code>NaN</code>, <code>Inf</code>, or unhandled values in estimator inputs.</li>
    <li><strong>Statistical Completeness:</strong> Summary tables contain zero <code>NaN</code> values for standard deviations or confidence intervals across $N=5$.</li>
    <li><strong>Convergence Transparency:</strong> Explicit logging of LR convergence status without overstating `n_iter_`.</li>
  </ol>
</div>

<div class="callout callout-gate">
  <strong>Execution Confirmation Gate:</strong><br>
  Per your instructions (<em>"Msg before ur starting nxt such that i will let u know whether the committed all the updates or not any"</em>), implementation will begin once you confirm this plan and verify that all remote and local updates on your side are fully committed.
</div>

</body>
</html>
"""

HTML_PATH.write_text(html_content, encoding="utf-8")
print(f"Written updated HTML plan to {HTML_PATH}")

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    str(HTML_PATH)
]

print("Compiling PDF with Microsoft Edge headless...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Returncode:", res.returncode)

if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
    print(f"SUCCESS: PDF generated at {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")
    shutil.copy2(PDF_PATH, r"C:\Users\ashis\Downloads\Part_B_Implementation_Plan.pdf")
    shutil.copy2(PDF_PATH, r"C:\Users\ashis\OneDrive\Desktop\Part_B_Implementation_Plan.pdf")
    print("Copied updated PDF to Downloads and Desktop successfully.")
else:
    print("FAILED: PDF not created or empty.")
