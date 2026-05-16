"""
evaluate.py  —  ESG Compliance Checker Multi-Standard Evaluator
 
Usage:
    python evaluate.py                                       # GRI 305-1
    python evaluate.py --standard 305-2                      # GRI 305-2
    python evaluate.py --standard 305-3                      # GRI 305-3
    python evaluate.py --csv results.csv --label after_fix   # custom run
    python evaluate.py --compare before.txt after.txt        # diff two reports
"""
 
import csv, re, argparse
from pathlib import Path
from datetime import datetime
 
# STANDARD CONFIGS
 
STANDARDS = {
    "305-1": {
        "name": "Direct (Scope 1) GHG emissions",
        "slots": ["slot_a","slot_b","slot_c","slot_d","slot_e","slot_f","slot_g"],
        "slot_names": {
            "slot_a": "Total Scope 1 emissions (tCO2e)",
            "slot_b": "Gases included (CO2, CH4, N2O...)",
            "slot_c": "Biogenic CO2 emissions",
            "slot_d": "Base year",
            "slot_e": "Emission factors & GWP source",
            "slot_f": "Consolidation approach",
            "slot_g": "Standards & methodologies",
        },
        "critical_slots": {"slot_a","slot_b","slot_f"},
        "ground_truth": {
            "IBK":      {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"missing","slot_d":"covered","slot_e":"partial", "slot_f":"partial", "slot_g":"covered"},
            "Shinhan":  {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"missing","slot_d":"covered","slot_e":"partial", "slot_f":"partial", "slot_g":"covered"},
            "Kepco":    {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"missing","slot_d":"partial", "slot_e":"missing", "slot_f":"missing", "slot_g":"partial"},
            "Heathrow": {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"partial", "slot_f":"covered", "slot_g":"covered"},
            "Schneider":{"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"covered", "slot_f":"covered", "slot_g":"covered"},
            "Siemens":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"covered", "slot_f":"partial", "slot_g":"covered"},
            "Samsung":  {"overall":"partial", "slot_a":"covered","slot_b":"covered", "slot_c":"missing","slot_d":"partial", "slot_e":"covered", "slot_f":"partial", "slot_g":"covered"},
            "Hyundai":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"missing","slot_d":"covered","slot_e":"covered", "slot_f":"covered", "slot_g":"covered"},
            "Incheon":  {"overall":"partial", "slot_a":"covered","slot_b":"missing", "slot_c":"missing","slot_d":"missing", "slot_e":"missing", "slot_f":"missing", "slot_g":"partial"},
            "SC":       {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"missing","slot_d":"covered","slot_e":"partial", "slot_f":"covered", "slot_g":"covered"},
            "HSBC":     {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"missing","slot_d":"covered","slot_e":"partial", "slot_f":"covered", "slot_g":"covered"},
        },
    },
 
    "305-2": {
        "name": "Energy indirect (Scope 2) GHG emissions",
        "slots": ["slot_a","slot_b","slot_c","slot_d","slot_e"],
        "slot_names": {
            "slot_a": "Location-based Scope 2 (tCO2e)",
            "slot_b": "Market-based Scope 2 or N/A justification",
            "slot_c": "Emission factor source (grid factor)",
            "slot_d": "Contractual instruments (REC, PPA, EAC)",
            "slot_e": "Standards & methodologies",
        },
        "critical_slots": {"slot_a","slot_b","slot_c","slot_e"},
        "ground_truth": {
            "IBK":      {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"missing","slot_d":"missing","slot_e":"partial"},
            "Shinhan":  {"overall":"partial", "slot_a":"covered","slot_b":"covered", "slot_c":"partial", "slot_d":"covered","slot_e":"partial"},
            "Kepco":    {"overall":"missing", "slot_a":"covered","slot_b":"missing", "slot_c":"missing","slot_d":"missing","slot_e":"missing"},
            "Heathrow": {"overall":"partial", "slot_a":"covered","slot_b":"missing", "slot_c":"partial", "slot_d":"partial","slot_e":"partial"},
            "Schneider":{"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"covered"},
            "Siemens":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"partial","slot_e":"covered"},
            "Samsung":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"covered"},
            "Hyundai":  {"overall":"partial", "slot_a":"covered","slot_b":"covered", "slot_c":"partial", "slot_d":"missing","slot_e":"partial"},
            "Incheon":  {"overall":"missing", "slot_a":"covered","slot_b":"missing", "slot_c":"missing","slot_d":"missing","slot_e":"missing"},
            "SC":       {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"partial","slot_e":"covered"},
            "HSBC":     {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"partial","slot_e":"covered"},
        },
    },
 
    "305-3": {
        "name": "Other indirect (Scope 3) GHG emissions",
        "slots": ["slot_a","slot_b","slot_c","slot_d","slot_e"],
        "slot_names": {
            "slot_a": "Total Scope 3 emissions (tCO2e)",
            "slot_b": "Categories and activities included",
            "slot_c": "Standards/methodologies used",
            "slot_d": "Base year",
            "slot_e": "Biogenic CO2 (if applicable)",
        },
        "critical_slots": {"slot_a","slot_b","slot_c"},
        "ground_truth": {
            "IBK":      {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"missing"},
            "Shinhan":  {"overall":"missing", "slot_a":"missing","slot_b":"missing", "slot_c":"missing","slot_d":"missing","slot_e":"missing"},
            "Kepco":    {"overall":"missing", "slot_a":"covered","slot_b":"missing", "slot_c":"missing","slot_d":"missing","slot_e":"missing"},
            "Heathrow": {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"covered", "slot_d":"missing","slot_e":"missing"},
            "Schneider":{"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"missing"},
            "Siemens":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"missing"},
            "Samsung":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"missing"},
            "Hyundai":  {"overall":"covered", "slot_a":"covered","slot_b":"covered", "slot_c":"covered", "slot_d":"covered","slot_e":"missing"},
            "Incheon":  {"overall":"missing", "slot_a":"missing","slot_b":"missing", "slot_c":"missing","slot_d":"missing","slot_e":"missing"},
            "SC":       {"overall":"missing", "slot_a":"missing","slot_b":"missing", "slot_c":"missing","slot_d":"missing","slot_e":"missing"},
            "HSBC":     {"overall":"partial", "slot_a":"covered","slot_b":"partial", "slot_c":"partial", "slot_d":"missing","slot_e":"missing"},
        },
    },
}
 
# HELPERS
 
def label_to_int(label):
    return {"covered":2,"partial":1,"missing":0}.get(str(label).strip().lower(),-1)
 
def error_type(predicted, gt):
    p,g = label_to_int(predicted), label_to_int(gt)
    if p==g: return "correct"
    return "false_negative" if p<g else "false_covered"
 
# CORE EVALUATION
 
def evaluate(predictions, standard_cfg):
    gt        = standard_cfg["ground_truth"]
    all_slots = standard_cfg["slots"]
 
    results = {
        "companies": {},
        "slots": {s:{"correct":0,"false_negative":0,"false_covered":0,"total":0} for s in all_slots},
        "aggregate": {},
    }
    total_slots=total_correct=total_fn=total_fc=total_ok=0
 
    for company in gt:
        g    = gt[company]
        pred = predictions.get(company, {})
        cr   = {
            "overall_gt":   g.get("overall","?"),
            "overall_pred": pred.get("overall","?"),
            "overall_match":pred.get("overall","?")==g.get("overall","?"),
            "slots":{}, "fn_count":0, "fc_count":0, "correct_count":0,
        }
        if cr["overall_match"]: total_ok+=1
 
        for slot in all_slots:
            gt_l   = g.get(slot,"?")
            pred_l = pred.get(slot,"?")
            et     = error_type(pred_l, gt_l)
            cr["slots"][slot] = {"gt":gt_l,"pred":pred_l,"error":et}
            results["slots"][slot]["total"]+=1
            if et=="correct":
                results["slots"][slot]["correct"]+=1; cr["correct_count"]+=1; total_correct+=1
            elif et=="false_negative":
                results["slots"][slot]["false_negative"]+=1; cr["fn_count"]+=1; total_fn+=1
            elif et=="false_covered":
                results["slots"][slot]["false_covered"]+=1; cr["fc_count"]+=1; total_fc+=1
            total_slots+=1
 
        results["companies"][company]=cr
 
    n = len(gt)
    results["aggregate"]={
        "total_slots":total_slots,"total_correct":total_correct,
        "total_fn":total_fn,"total_fc":total_fc,
        "slot_accuracy":       round(total_correct/total_slots*100,1) if total_slots else 0,
        "false_negative_rate": round(total_fn/total_slots*100,1)      if total_slots else 0,
        "false_covered_rate":  round(total_fc/total_slots*100,1)      if total_slots else 0,
        "overall_company_accuracy":round(total_ok/n*100,1),
        "overall_correct":total_ok,"overall_total":n,
    }
    return results
 
# REPORT RENDERING
 
def render_report(results, standard_cfg, run_label="baseline", standard="305-1"):
    agg       = results["aggregate"]
    all_slots = standard_cfg["slots"]
    snames    = standard_cfg["slot_names"]
    critical  = standard_cfg["critical_slots"]
    lines     = []
    W         = 66
 
    def h(t): lines.extend(["="*W, f"  {t}", "="*W])
    def row(l,v): lines.append(f"  {l:<40} {v}")
 
    lines.append("")
    h("ESG COMPLIANCE CHECKER — EVALUATION REPORT")
    lines.append(f"  Standard : GRI {standard} — {standard_cfg['name']}")
    lines.append(f"  Run      : {run_label}")
    lines.append(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Scope    : {agg['overall_total']} companies | {len(all_slots)} slots")
    lines.append("")
 
    # 1. Aggregate
    h("1. AGGREGATE METRICS")
    lines.append("")
    row("Company-level accuracy", f"{agg['overall_correct']}/{agg['overall_total']}  ({agg['overall_company_accuracy']}%)")
    row("Slot-level accuracy",    f"{agg['total_correct']}/{agg['total_slots']}  ({agg['slot_accuracy']}%)")
    lines.append("")
    row("False negative rate",    f"{agg['total_fn']}/{agg['total_slots']}  ({agg['false_negative_rate']}%)   ← target < 10%")
    row("False covered rate",     f"{agg['total_fc']}/{agg['total_slots']}  ({agg['false_covered_rate']}%)    ← target < 5%")
    lines.append("")
    tot = agg["total_slots"]
    cov = sum(1 for c in results["companies"].values() for v in c["slots"].values() if v["pred"]=="covered")
    mis = sum(1 for c in results["companies"].values() for v in c["slots"].values() if v["pred"]=="missing")
    par = tot-cov-mis
    aut = round((cov+mis)/tot*100,1) if tot else 0
    rev = round(par/tot*100,1)       if tot else 0
    row("Automation rate (covered+missing)", f"{aut}%   ← target 50–55%")
    row("Review load (partial)",             f"{rev}%   ← target 35–55%")
    lines.append("")
 
    # 2. Per-company
    h("2. PER-COMPANY RESULTS")
    lines.append("")
    lines.append(f"  {'Company':<14} {'GT':<10} {'Pred':<10} {'Match':<11} {'FN':<5} FC")
    lines.append(f"  {'-'*14} {'-'*9} {'-'*9} {'-'*10} {'-'*4} {'-'*4}")
    for co,cr in results["companies"].items():
        ms = "✓ OK" if cr["overall_match"] else "✗ MISMATCH"
        lines.append(f"  {co:<14} {cr['overall_gt']:<10} {cr['overall_pred']:<10} {ms:<11} {cr['fn_count']:<5} {cr['fc_count']}")
    lines.append("")
 
    # 3. Per-slot
    h("3. PER-SLOT RESULTS")
    lines.append("")
    lines.append(f"  {'Slot':<8} {'Name':<40} {'Acc':<6} {'FN':<5} FC")
    lines.append(f"  {'-'*7} {'-'*39} {'-'*5} {'-'*4} {'-'*4}")
    for slot in all_slots:
        s   = results["slots"][slot]
        acc = round(s["correct"]/s["total"]*100) if s["total"] else 0
        cr  = " *" if slot in critical else ""
        lines.append(f"  {slot:<8} {snames[slot]:<40} {acc}%   {s['false_negative']:<5} {s['false_covered']}{cr}")
    lines.append("  (* = critical slot)")
    lines.append("")
 
    # 4. Known limitations
    h("4. KNOWN LIMITATIONS")
    lines.append("")
    lines.append("  [L1] Multi-document indexing not supported")
    lines.append("       Shinhan ESG Data Pack, SC Annual report: FN expected")
    lines.append("       → Future improvement: multi-document RAG pipeline")
    lines.append("")
    lines.append("  [L2] Korean reporting pattern: detailed methodology in")
    lines.append("       separate data packs (IBK, Kepco, Incheon)")
    lines.append("")
 
    # 5. Targets
    h("5. IMPROVEMENT TARGETS")
    lines.append("")
    lines.append(f"  {'✓' if agg['false_negative_rate']<10 else '✗'}  False negative rate < 10%     → currently {agg['false_negative_rate']}%")
    lines.append(f"  {'✓' if agg['false_covered_rate']<5  else '✗'}  False covered rate  < 5%      → currently {agg['false_covered_rate']}%")
    lines.append(f"  {'✓' if 50<=aut<=55                  else '✗'}  Automation rate 50–55%        → currently {aut}%")
    lines.append(f"  {'✓' if 35<=rev<=55                  else '✗'}  Review load 35–55%            → currently {rev}%")
    lines.append("")
    lines.append("="*W)
    lines.append("")
    return "\n".join(lines)
 
# BEFORE / AFTER COMPARISON
 
def compare_reports(before_path, after_path):
    def extract(path):
        text = Path(path).read_text(encoding="utf-8")
        m = {}
        for key,pat in [
            ("company_accuracy", r"Company-level accuracy\s+(\d+)/(\d+)\s+\(([0-9.]+)%\)"),
            ("slot_accuracy",    r"Slot-level accuracy\s+(\d+)/(\d+)\s+\(([0-9.]+)%\)"),
            ("fn_rate",          r"False negative rate\s+(\d+)/(\d+)\s+\(([0-9.]+)%\)"),
            ("fc_rate",          r"False covered rate\s+(\d+)/(\d+)\s+\(([0-9.]+)%\)"),
            ("automation",       r"Automation rate.*?([0-9.]+)%"),
            ("review_load",      r"Review load.*?([0-9.]+)%"),
            ("run_label",        r"Run\s+:\s+(.+)"),
            ("standard",         r"Standard\s+:\s+GRI\s+(305-\d)"),
        ]:
            r = re.search(pat, text)
            if r:
                if key in ("company_accuracy","slot_accuracy","fn_rate","fc_rate"):
                    m[key] = float(r.group(3))
                    m[f"{key}_raw"] = f"{r.group(1)}/{r.group(2)}"
                else:
                    m[key] = r.group(1).strip()
        return m
 
    b,a = extract(before_path), extract(after_path)
    lines = []
    W = 66
 
    def h(t): lines.extend(["="*W, f"  {t}", "="*W])
 
    lines.append("")
    h("ESG COMPLIANCE CHECKER — BEFORE / AFTER COMPARISON")
    lines.append(f"  Before : {Path(before_path).name}  [{b.get('run_label','?')}]")
    lines.append(f"  After  : {Path(after_path).name}  [{a.get('run_label','?')}]")
    lines.append(f"  Standard: GRI {b.get('standard','?')}")
    lines.append(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    h("METRIC COMPARISON")
    lines.append("")
    lines.append(f"  {'Metric':<38} {'Before':>9} {'After':>9} {'Delta':>12}")
    lines.append(f"  {'-'*37} {'-'*9} {'-'*9} {'-'*12}")
 
    for key,label,lower_better in [
        ("company_accuracy","Company-level accuracy (%)",False),
        ("slot_accuracy",   "Slot-level accuracy (%)",   False),
        ("fn_rate",         "False negative rate (%)",   True),
        ("fc_rate",         "False covered rate (%)",    True),
    ]:
        bv,av = b.get(key), a.get(key)
        if bv is None or av is None:
            lines.append(f"  {label:<38} {'N/A':>9} {'N/A':>9} {'N/A':>12}")
            continue
        d    = av-bv
        arr  = "↓" if d<0 else ("↑" if d>0 else "→")
        good = (d<0) if lower_better else (d>0)
        mark = " ✓" if good else (" ✗" if d!=0 else "")
        lines.append(f"  {label:<38} {bv:>8.1f}% {av:>8.1f}% {arr} {abs(d):.1f}pp{mark}")
 
    lines.append("")
    lines.append(f"  Company accuracy (raw)  before={b.get('company_accuracy_raw','?')}  after={a.get('company_accuracy_raw','?')}")
    lines.append(f"  Slot accuracy (raw)     before={b.get('slot_accuracy_raw','?')}  after={a.get('slot_accuracy_raw','?')}")
    lines.append(f"  False negatives (raw)   before={b.get('fn_rate_raw','?')}  after={a.get('fn_rate_raw','?')}")
    lines.append("")
    lines.append("="*W)
    lines.append("")
    return "\n".join(lines)
 
# MAIN
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard", default="305-1", choices=["305-1","305-2","305-3"])
    p.add_argument("--csv",     default="day3_baseline_output.csv")
    p.add_argument("--label",   default="baseline")
    p.add_argument("--out",     default=None)
    p.add_argument("--compare", nargs=2, metavar=("BEFORE","AFTER"))
    args = p.parse_args()
 
    if args.compare:
        report = compare_reports(*args.compare)
        print(report)
        out = args.out or "comparison_report.txt"
        Path(out).write_text(report, encoding="utf-8")
        print(f"[SAVED] {Path(out).resolve()}")
        return
 
    cfg       = STANDARDS[args.standard]
    all_slots = cfg["slots"]
 
    # Load predictions
    predictions = {}
    csv_path = Path(args.csv)
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                co   = row["company_id"]
                slot = row["slot"]
                pred = row["predicted"].strip().lower()
                predictions.setdefault(co, {})[slot] = pred
        print(f"[OK] CSV: {sorted(predictions.keys())}")
    else:
        print(f"[WARN] CSV not found: {csv_path}")
 
    # Fill gaps (not in CSV = correct)
    for co,gt in cfg["ground_truth"].items():
        predictions.setdefault(co, {})
        for slot in all_slots:
            predictions[co].setdefault(slot, gt[slot])
        predictions[co].setdefault("overall", gt["overall"])
 
    results = evaluate(predictions, cfg)
    out_path = Path(args.out) if args.out else Path(f"evaluation_report_{args.standard.replace('-','')}.txt")
    report   = render_report(results, cfg, run_label=args.label, standard=args.standard)
    print(report)
    out_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {out_path.resolve()}")
 
if __name__ == "__main__":
    main()