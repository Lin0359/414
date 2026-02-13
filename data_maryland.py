import requests
import pandas as pd
import matplotlib.pyplot as plt


YEARS = [2012, 2014, 2016, 2018, 2022]

STATE_FIPS = "24"  # Maryland


VAR_INCOME = "B19013_001E"   # Median household income
VAR_GINI   = "B19083_001E"   # Gini index of income inequality

def fetch_acs_county_year(year: int) -> pd.DataFrame:
    """
    Fetch ACS 5-year county data for Maryland for a given year.
    Returns a cleaned DataFrame with columns: NAME, year, income, gini, state, county
    """
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{VAR_INCOME},{VAR_GINI}",
        "for": "county:*",
        "in": f"state:{STATE_FIPS}",
    }
    headers = {"User-Agent": "Mozilla/5.0 (data-collection for coursework)"}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    df["year"] = year
    df["income"] = pd.to_numeric(df[VAR_INCOME], errors="coerce")
    df["gini"]   = pd.to_numeric(df[VAR_GINI], errors="coerce")


    df = df.dropna(subset=["income", "gini"]).copy()

    return df[["NAME", "state", "county", "year", "income", "gini"]]


all_dfs = []
failed_years = []

for y in YEARS:
    try:
        df_y = fetch_acs_county_year(y)
        all_dfs.append(df_y)
        print(f"[OK] {y}: rows={len(df_y)}")
    except Exception as e:
        failed_years.append((y, str(e)))
        print(f"[FAIL] {y}: {e}")

if not all_dfs:
    raise RuntimeError("No data fetched. Check network / years / API availability.")

final_df = pd.concat(all_dfs, ignore_index=True)


state_trend = (
    final_df.groupby("year")
    .agg(
        avg_income=("income", "mean"),
        median_income=("income", "median"),
        avg_gini=("gini", "mean"),
        median_gini=("gini", "median"),
        n_counties=("NAME", "count"),
    )
    .reset_index()
    .sort_values("year")
)

print("\n=== Maryland statewide (county-average) trend ===")
print(state_trend.to_string(index=False))


corr_by_year = (
    final_df.groupby("year")
    .apply(lambda g: g["income"].corr(g["gini"]))
    .reset_index(name="corr_income_gini")
    .sort_values("year")
)

print("\n=== Correlation (county-level): income vs gini ===")
print(corr_by_year.to_string(index=False))


overall_corr = final_df["income"].corr(final_df["gini"])
print(f"\n=== Overall correlation (all counties, all selected years): {overall_corr:.4f} ===")


plt.figure()
plt.plot(state_trend["year"], state_trend["avg_income"])
plt.title("Maryland Avg County Median Household Income (ACS 5-year)")
plt.xlabel("Year")
plt.ylabel("Income (USD)")
plt.show()

plt.figure()
plt.plot(state_trend["year"], state_trend["avg_gini"])
plt.title("Maryland Avg County Gini Index (ACS 5-year)")
plt.xlabel("Year")
plt.ylabel("Gini Index")
plt.show()


if failed_years:
    print("\n=== Failed years ===")
    for y, msg in failed_years:
        print(f"{y}: {msg}")
