"""
E-Commerce Intelligence Platform — A/B Testing Framework
=========================================================
Tests: Two-sample t-test, z-test for proportions, Chi-square
Metrics: Conversion rate, Revenue per user, AOV

Run: python ml/ab_testing.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

ALPHA = 0.05  # significance level


# ── SAMPLE DATA GENERATOR (replace with real experiment data) ─────────────────
def generate_experiment_data(n_control=5000, n_test=5000, seed=42):
    """
    Simulate A/B test: Control = baseline, Test = new recommendation engine
    In production: join with your experiment_assignments table from S3/Athena
    """
    rng = np.random.default_rng(seed)
    control = pd.DataFrame({
        "variant":    "control",
        "user_id":    range(n_control),
        "converted":  rng.binomial(1, 0.12, n_control),        # 12% base CVR
        "revenue":    rng.lognormal(4.0, 1.2, n_control),
        "session_pages": rng.poisson(3.5, n_control),
    })
    test = pd.DataFrame({
        "variant":    "test",
        "user_id":    range(n_control, n_control + n_test),
        "converted":  rng.binomial(1, 0.138, n_test),           # 13.8% test CVR (+15%)
        "revenue":    rng.lognormal(4.1, 1.2, n_test),
        "session_pages": rng.poisson(4.1, n_test),
    })
    return pd.concat([control, test], ignore_index=True)


# ── METRIC COMPUTATION ────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute key business metrics per variant."""
    metrics = df.groupby("variant").agg(
        users         = ("user_id",    "count"),
        conversions   = ("converted",  "sum"),
        total_revenue = ("revenue",    "sum"),
        avg_revenue   = ("revenue",    "mean"),
    ).reset_index()
    metrics["conversion_rate"] = metrics["conversions"] / metrics["users"]
    metrics["revenue_per_user"]= metrics["total_revenue"] / metrics["users"]
    log.info("=== METRIC SUMMARY ===")
    log.info(metrics[["variant","users","conversions","conversion_rate",
                       "avg_revenue","revenue_per_user"]].round(4).to_string(index=False))
    return metrics


# ── STATISTICAL TESTS ─────────────────────────────────────────────────────────
def test_conversion_rate(df: pd.DataFrame) -> dict:
    """Z-test for difference in conversion rates (proportions)."""
    ctrl = df[df["variant"]=="control"]
    test = df[df["variant"]=="test"]

    n1, x1 = len(ctrl), ctrl["converted"].sum()
    n2, x2 = len(test), test["converted"].sum()
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1+x2)/(n1+n2)

    se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z  = (p2-p1)/se
    p_val = 2*(1-stats.norm.cdf(abs(z)))
    lift  = (p2-p1)/p1*100

    result = {
        "test": "Z-test (conversion rate)",
        "control_cvr": round(p1,4), "test_cvr": round(p2,4),
        "lift_pct": round(lift,2), "z_stat": round(z,4),
        "p_value": round(p_val,6), "significant": p_val < ALPHA,
        "decision": "REJECT H₀ — significant difference" if p_val < ALPHA
                    else "FAIL TO REJECT H₀ — no significant difference"
    }
    return result


def test_revenue(df: pd.DataFrame) -> dict:
    """Two-sample t-test for revenue per user."""
    ctrl_rev = df[df["variant"]=="control"]["revenue"]
    test_rev = df[df["variant"]=="test"]["revenue"]

    t, p_val = stats.ttest_ind(ctrl_rev, test_rev, equal_var=False)  # Welch's t-test
    lift = (test_rev.mean()-ctrl_rev.mean())/ctrl_rev.mean()*100

    result = {
        "test": "Welch t-test (revenue per user)",
        "control_avg": round(ctrl_rev.mean(),2),
        "test_avg":    round(test_rev.mean(),2),
        "lift_pct":    round(lift,2),
        "t_stat":      round(t,4),
        "p_value":     round(p_val,6),
        "significant": p_val < ALPHA,
        "decision": "REJECT H₀ — significant revenue difference" if p_val < ALPHA
                    else "FAIL TO REJECT H₀ — no significant revenue difference"
    }
    return result


def test_session_pages(df: pd.DataFrame) -> dict:
    """Mann-Whitney U test for session pages (non-parametric)."""
    ctrl = df[df["variant"]=="control"]["session_pages"]
    test = df[df["variant"]=="test"]["session_pages"]
    u, p_val = stats.mannwhitneyu(ctrl, test, alternative="two-sided")
    result = {
        "test": "Mann-Whitney U (session pages)",
        "control_median": ctrl.median(), "test_median": test.median(),
        "u_stat": round(u,2), "p_value": round(p_val,6),
        "significant": p_val < ALPHA,
        "decision": "REJECT H₀" if p_val < ALPHA else "FAIL TO REJECT H₀"
    }
    return result


# ── SAMPLE SIZE CALCULATOR ─────────────────────────────────────────────────────
def required_sample_size(baseline_cvr=0.12, min_detectable_effect=0.02,
                          alpha=0.05, power=0.8):
    """Calculate minimum sample size per variant."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta  = stats.norm.ppf(power)
    p1, p2  = baseline_cvr, baseline_cvr + min_detectable_effect
    p_avg   = (p1+p2)/2
    n = (z_alpha*np.sqrt(2*p_avg*(1-p_avg)) + z_beta*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2 \
        / (p2-p1)**2
    return int(np.ceil(n))


# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def plot_results(df, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    colors = {"control":"#4472C4","test":"#ED7D31"}

    # Conversion rate
    cvr = metrics.set_index("variant")["conversion_rate"]
    cvr.plot(kind="bar", ax=axes[0], color=[colors[v] for v in cvr.index], rot=0)
    axes[0].set_title("Conversion Rate"); axes[0].set_ylabel("CVR")
    for bar, v in zip(axes[0].patches, cvr):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                     f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")

    # Revenue per user
    rpu = metrics.set_index("variant")["revenue_per_user"]
    rpu.plot(kind="bar", ax=axes[1], color=[colors[v] for v in rpu.index], rot=0)
    axes[1].set_title("Revenue per User"); axes[1].set_ylabel("$")

    # Revenue distribution
    for var, grp in df.groupby("variant"):
        axes[2].hist(grp["revenue"].clip(upper=grp["revenue"].quantile(0.99)),
                     bins=50, alpha=0.6, label=var, color=colors[var])
    axes[2].set_title("Revenue Distribution"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("dashboard/ab_test_results.png", dpi=150)
    log.info("A/B test plot saved → dashboard/ab_test_results.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Generating experiment data...")
    df = generate_experiment_data()

    log.info("Computing metrics...")
    metrics = compute_metrics(df)

    log.info("\n=== STATISTICAL TESTS ===")
    for fn in [test_conversion_rate, test_revenue, test_session_pages]:
        res = fn(df)
        log.info(f"\n{res['test']}")
        for k,v in res.items():
            if k != "test":
                log.info(f"  {k}: {v}")

    n = required_sample_size()
    log.info(f"\nRequired sample size per variant (12%→14%, power=80%): {n:,}")

    plot_results(df, metrics)
    metrics.to_csv("dashboard/ab_test_metrics.csv", index=False)
    log.info("🎉 A/B testing complete!")
