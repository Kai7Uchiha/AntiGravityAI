import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path_or_url: str) -> pd.DataFrame:
    """Load CSV from local path or URL"""
    if file_path_or_url.startswith("http"):
        return pd.read_csv(file_path_or_url)
    else:
        return pd.read_csv(file_path_or_url)

def generate_profile(df: pd.DataFrame, title: str = "EDA Report") -> str:
    """Create a simple text-based profile report"""
    filename = f"{title.replace(' ', '_')}.txt"
    
    with open(filename, "w") as f:
        f.write(f"=== {title} ===\n\n")
        f.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        f.write("=== Column Information ===\n")
        for col in df.columns:
            f.write(f"\n{col}:\n")
            f.write(f"  Type: {df[col].dtype}\n")
            f.write(f"  Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)\n")
            f.write(f"  Unique values: {df[col].nunique()}\n")
            
            if df[col].dtype in ["int64", "float64"]:
                f.write(f"  Min: {df[col].min()}\n")
                f.write(f"  Max: {df[col].max()}\n")
                f.write(f"  Mean: {df[col].mean():.2f}\n")
                f.write(f"  Median: {df[col].median():.2f}\n")
            else:
                top_value = df[col].value_counts().index[0]
                top_count = df[col].value_counts().values[0]
                f.write(f"  Most common: {top_value} ({top_count} occurrences)\n")
        
        f.write("\n=== Missing Values Summary ===\n")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                f.write(f"  {col}: {df[col].isnull().sum()} missing\n")
        else:
            f.write("  No missing values found!\n")
        
        f.write("\n=== Duplicate Rows ===\n")
        f.write(f"  Duplicate rows: {df.duplicated().sum()}\n")
    
    print(f"   ✅ Profile report saved: {filename}")
    return filename

def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-clean dataset: remove duplicates, fill missing values"""
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                replacement = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(replacement)
    return df

def quick_plot(df: pd.DataFrame, col: str, save_path: str = None):
    """Create and optionally save a plot"""
    plt.figure(figsize=(6, 4))
    if df[col].dtype in ["int64", "float64"]:
        sns.histplot(df[col], kde=True)
    else:
        df[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"   ✅ Plot saved: {save_path}")
    else:
        plt.show()
    plt.close()