#!/usr/bin/env python3
"""
EXTENDED TEMPLATE TEST: 50+ templates to see if G₂ is genuinely special
"""

import numpy as np
np.random.seed(42)

EXPERIMENTAL = 137.036
G2_nums = [14, 12, 13, 2, 10, 6, 7]  # dim, |Δ|, |Δ|+1, rank, dim-4, |Δ|/2, dim/2

print("=" * 90)
print("EXTENDED TEMPLATE TEST: 50+ TEMPLATES")
print("=" * 90)

def test_template(template_fn, n_random=1000):
    """Test G₂ vs random for a template"""
    # G₂ best
    g2_best = None
    g2_best_error = float('inf')
    g2_params = None
    
    for a in G2_nums:
        for b in G2_nums:
            for c in G2_nums:
                for d in G2_nums:
                    try:
                        result = template_fn(a, b, c, d)
                        if result is not None and 0 < result < 10000:
                            error = abs(result - EXPERIMENTAL)
                            if error < g2_best_error:
                                g2_best_error = error
                                g2_best = result
                                g2_params = (a, b, c, d)
                    except:
                        pass
    
    if g2_best is None:
        return None
    
    # Random test
    better_count = 0
    valid_count = 0
    for _ in range(n_random):
        nums = np.random.randint(2, 20, 7)
        best_err = float('inf')
        for a in nums:
            for b in nums:
                for c in nums:
                    for d in nums:
                        try:
                            result = template_fn(a, b, c, d)
                            if result is not None and 0 < result < 10000:
                                err = abs(result - EXPERIMENTAL)
                                if err < best_err:
                                    best_err = err
                        except:
                            pass
        if best_err < float('inf'):
            valid_count += 1
            if best_err < g2_best_error:
                better_count += 1
    
    percentile = 100 * (1 - better_count / valid_count) if valid_count > 0 else 0
    return g2_best, g2_best_error, g2_params, percentile, valid_count

# Define many templates
templates = {
    # Duality-type
    "1/x + ab*x = cd*π²": lambda a,b,c,d: (1/((c*d*np.pi**2 - np.sqrt((c*d*np.pi**2)**2 - 4*a*b))/(2*a*b))) if (c*d*np.pi**2)**2 > 4*a*b and a*b > 0 else None,
    "1/x + a*x = b*π²": lambda a,b,c,d: (1/((b*np.pi**2 - np.sqrt((b*np.pi**2)**2 - 4*a))/(2*a))) if (b*np.pi**2)**2 > 4*a and a > 0 else None,
    "1/x + ab*x = c*π": lambda a,b,c,d: (1/((c*np.pi - np.sqrt((c*np.pi)**2 - 4*a*b))/(2*a*b))) if (c*np.pi)**2 > 4*a*b and a*b > 0 else None,
    
    # Polynomial
    "a*b + c*d": lambda a,b,c,d: a*b + c*d,
    "a*b*c/d": lambda a,b,c,d: a*b*c/d if d != 0 else None,
    "a*b - c*d": lambda a,b,c,d: a*b - c*d if a*b > c*d else None,
    "(a+b)*(c+d)/2": lambda a,b,c,d: (a+b)*(c+d)/2,
    "a² + b² + c": lambda a,b,c,d: a**2 + b**2 + c,
    "a² - b² + c*d": lambda a,b,c,d: a**2 - b**2 + c*d if a > b else None,
    "(a*b)² / (c*d)": lambda a,b,c,d: (a*b)**2 / (c*d) if c*d != 0 else None,
    "a³/b + c": lambda a,b,c,d: a**3/b + c if b != 0 else None,
    
    # With π
    "a*b*π/c": lambda a,b,c,d: a*b*np.pi/c if c != 0 else None,
    "a*π² + b": lambda a,b,c,d: a*np.pi**2 + b,
    "a*π² - b*c": lambda a,b,c,d: a*np.pi**2 - b*c if a*np.pi**2 > b*c else None,
    "(a+b)*π²/c": lambda a,b,c,d: (a+b)*np.pi**2/c if c != 0 else None,
    "a*b/(c*π)": lambda a,b,c,d: a*b/(c*np.pi) if c != 0 else None,
    "a*π + b*π + c": lambda a,b,c,d: a*np.pi + b*np.pi + c,
    "a*b*c*π²/d²": lambda a,b,c,d: a*b*c*np.pi**2/d**2 if d != 0 else None,
    
    # Exponential
    "e^(a/b) + c": lambda a,b,c,d: np.exp(a/b) + c if b != 0 and a/b < 10 else None,
    "e^(a/b) - c": lambda a,b,c,d: np.exp(a/b) - c if b != 0 and a/b < 10 else None,
    "a*e^(b/c)": lambda a,b,c,d: a*np.exp(b/c) if c != 0 and b/c < 10 else None,
    "e^a / b": lambda a,b,c,d: np.exp(a) / b if a < 10 and b != 0 else None,
    "c*e^(a/b)/d": lambda a,b,c,d: c*np.exp(a/b)/d if b != 0 and d != 0 and a/b < 10 else None,
    
    # Logarithmic  
    "a*ln(b) + c": lambda a,b,c,d: a*np.log(b) + c if b > 0 else None,
    "a*ln(b*c)": lambda a,b,c,d: a*np.log(b*c) if b*c > 0 else None,
    "a*ln(b) + c*ln(d)": lambda a,b,c,d: a*np.log(b) + c*np.log(d) if b > 0 and d > 0 else None,
    "a*b*ln(c)": lambda a,b,c,d: a*b*np.log(c) if c > 0 else None,
    "a²*ln(b)/c": lambda a,b,c,d: a**2*np.log(b)/c if b > 0 and c != 0 else None,
    
    # Square roots
    "a*√b + c*d": lambda a,b,c,d: a*np.sqrt(b) + c*d if b >= 0 else None,
    "√(a*b*c*d)": lambda a,b,c,d: np.sqrt(a*b*c*d) if a*b*c*d >= 0 else None,
    "a*√(b*c) + d": lambda a,b,c,d: a*np.sqrt(b*c) + d if b*c >= 0 else None,
    "(a+b)*√c": lambda a,b,c,d: (a+b)*np.sqrt(c) if c >= 0 else None,
    "√(a²+b²)*c": lambda a,b,c,d: np.sqrt(a**2+b**2)*c,
    
    # Trigonometric (using π multiples)
    "a/sin(π/b) + c": lambda a,b,c,d: a/np.sin(np.pi/b) + c if b != 0 and abs(np.sin(np.pi/b)) > 0.01 else None,
    "a/tan(π/b)": lambda a,b,c,d: a/np.tan(np.pi/b) if b != 0 and abs(np.tan(np.pi/b)) > 0.01 else None,
    
    # Rational
    "(a²+b²)/(c-d)": lambda a,b,c,d: (a**2+b**2)/(c-d) if c != d else None,
    "a*b/(c-d)": lambda a,b,c,d: a*b/(c-d) if c != d and a*b/(c-d) > 0 else None,
    "(a+b+c)/d * 10": lambda a,b,c,d: (a+b+c)/d * 10 if d != 0 else None,
    "a*(b+c+d)": lambda a,b,c,d: a*(b+c+d),
    
    # Mixed
    "a*π*b/ln(c)": lambda a,b,c,d: a*np.pi*b/np.log(c) if c > 1 else None,
    "e^(a/b)*c/d": lambda a,b,c,d: np.exp(a/b)*c/d if b != 0 and d != 0 and a/b < 10 else None,
    "a*π²/ln(b*c)": lambda a,b,c,d: a*np.pi**2/np.log(b*c) if b*c > 1 else None,
    "√(a*b)*π + c": lambda a,b,c,d: np.sqrt(a*b)*np.pi + c if a*b >= 0 else None,
    "(a*b)^(c/d)": lambda a,b,c,d: (a*b)**(c/d) if a*b > 0 and d != 0 and 0 < c/d < 5 else None,
    
    # More duality variants
    "1/x + a*x = b*c": lambda a,b,c,d: (1/((b*c - np.sqrt((b*c)**2 - 4*a))/(2*a))) if (b*c)**2 > 4*a and a > 0 else None,
    "1/x + a*x = b + c*π": lambda a,b,c,d: (1/((b+c*np.pi - np.sqrt((b+c*np.pi)**2 - 4*a))/(2*a))) if (b+c*np.pi)**2 > 4*a and a > 0 else None,
    "1/x² + a*x = b": lambda a,b,c,d: None,  # cubic, skip for now
    
    # Power laws
    "a^b / c": lambda a,b,c,d: a**b / c if c != 0 and b < 10 else None,
    "a^(b/c) * d": lambda a,b,c,d: a**(b/c) * d if c != 0 and 0 < b/c < 5 else None,
}

print(f"\nTesting {len(templates)} templates...")
print("=" * 90)

results = []

for name, fn in templates.items():
    result = test_template(fn, n_random=500)
    if result:
        g2_pred, g2_err, g2_params, percentile, n_valid = result
        results.append({
            'name': name,
            'prediction': g2_pred,
            'error': g2_err,
            'params': g2_params,
            'percentile': percentile
        })

# Sort by error
results.sort(key=lambda x: x['error'])

print(f"\nTested {len(results)} valid templates")
print("\n" + "=" * 90)
print("TOP 20 TEMPLATES (sorted by error)")
print("=" * 90)
print(f"{'Rank':<5} {'Template':<30} {'Prediction':<12} {'Error':<10} {'%ile':<8} {'Verdict'}")
print("-" * 90)

for i, r in enumerate(results[:20]):
    verdict = "SPECIAL" if r['percentile'] > 95 else "GOOD" if r['percentile'] > 80 else "AVG" if r['percentile'] > 50 else "POOR"
    print(f"{i+1:<5} {r['name']:<30} {r['prediction']:<12.4f} {r['error']:<10.4f} {r['percentile']:<8.1f} {verdict}")

# Statistics
print("\n" + "=" * 90)
print("STATISTICS")
print("=" * 90)

special = [r for r in results if r['percentile'] > 95]
good = [r for r in results if r['percentile'] > 80]
avg = [r for r in results if 50 < r['percentile'] <= 80]
poor = [r for r in results if r['percentile'] <= 50]

print(f"SPECIAL (>95%): {len(special)}/{len(results)} = {100*len(special)/len(results):.1f}%")
print(f"GOOD (80-95%):  {len(good)-len(special)}/{len(results)} = {100*(len(good)-len(special))/len(results):.1f}%")
print(f"AVERAGE (50-80%): {len(avg)}/{len(results)} = {100*len(avg)/len(results):.1f}%")
print(f"POOR (<50%):    {len(poor)}/{len(results)} = {100*len(poor)/len(results):.1f}%")

if special:
    print(f"\nSPECIAL templates:")
    for r in special:
        print(f"  {r['name']}: prediction={r['prediction']:.4f}, error={r['error']:.4f}")

# Expected by chance
print("\n" + "=" * 90)
print("EXPECTED BY CHANCE")
print("=" * 90)
print(f"""
If G₂ numbers were random, we'd expect:
  ~5% of templates to be "SPECIAL" by chance = {0.05*len(results):.1f} templates
  
Actual SPECIAL templates: {len(special)}

Chi-square-like comparison:
  Expected: {0.05*len(results):.1f}
  Observed: {len(special)}
  Ratio: {len(special)/(0.05*len(results)):.2f}x expected
""")

if len(special) > 2 * 0.05 * len(results):
    print("CONCLUSION: G₂ appears MORE special than random chance would predict.")
elif len(special) < 0.5 * 0.05 * len(results):
    print("CONCLUSION: G₂ appears LESS special than random chance.")
else:
    print("CONCLUSION: G₂ performance is CONSISTENT with random chance.")
