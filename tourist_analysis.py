"""
Moduł analizy turystycznej - Etap 2
Wyszukiwarka popularnych miejsc turystycznych

Analiza sezonowości turystyki oraz identyfikacja popularnych destynacji
bazując na danych Eurostatu dotyczących noclegów.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Konfiguracja stylu wykresów
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Wczytuje dane z pliku CSV i przygotowuje je do analizy.
    
    Args:
        csv_path: Ścieżka do pliku CSV z danymi Eurostatu
        
    Returns:
        DataFrame z przygotowanymi danymi
    """
    print(f"Wczytywanie danych z {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Parsowanie TIME_PERIOD jako datetime
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y-%m')
    
    # Konwersja OBS_VALUE na typ numeryczny (usunięcie wartości nieprawidłowych)
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    
    print(f"Wczytano {len(df)} rekordów")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści dane: usuwa brakujące wartości i ogranicza do pełnych lat.
    
    Args:
        df: DataFrame z surowymi danymi
        
    Returns:
        DataFrame z oczyszczonymi danymi
    """
    print("\nCzyszczenie danych...")
    initial_count = len(df)
    
    # Usuwanie rekordów z brakującymi wartościami noclegów
    df_clean = df.dropna(subset=['OBS_VALUE']).copy()
    
    # Usuwanie wartości ujemnych (błędne dane)
    df_clean = df_clean[df_clean['OBS_VALUE'] >= 0]
    
    # Ekstrakcja roku i miesiąca
    df_clean['year'] = df_clean['TIME_PERIOD'].dt.year
    df_clean['month'] = df_clean['TIME_PERIOD'].dt.month
    
    # Identyfikacja pełnych lat (lata z danymi dla wszystkich 12 miesięcy)
    year_counts = df_clean.groupby('year').size()
    complete_years = year_counts[year_counts >= 12].index.tolist()
    
    if complete_years:
        # Ograniczenie do pełnych lat (zakres np. 2015-2024)
        min_year = min(complete_years)
        max_year = max(complete_years)
        print(f"Pełne lata dostępne w danych: {min_year} - {max_year}")
        df_clean = df_clean[df_clean['year'].isin(complete_years)]
    else:
        print("Uwaga: Brak pełnych lat w danych. Używam wszystkich dostępnych danych.")
    
    # Sprawdzenie liczby miesięcy na kraj na rok
    country_year_counts = df_clean.groupby(['geo', 'year']).size()
    valid_country_years = country_year_counts[country_year_counts == 12].index
    
    # # Filtrowanie do krajów z pełnymi latami
    # df_clean['year_geo'] = list(zip(df_clean['year'], df_clean['geo']))
    # df_clean = df_clean[df_clean['year_geo'].isin(valid_country_years)]
    # df_clean = df_clean.drop(columns=['year_geo'])
    #
    removed_count = initial_count - len(df_clean)
    print(f"Usunięto {removed_count} rekordów ({initial_count - removed_count} pozostało)")
    
    return df_clean


def analyze_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizuje sezonowość turystyki.
    
    Agreguje dane per kraj i per miesiąc (średnia wieloletnia),
    identyfikuje miesiące wysokiego i niskiego sezonu.
    
    Args:
        df: DataFrame z oczyszczonymi danymi
        
    Returns:
        DataFrame z analizą sezonowości
    """
    print("\nAnaliza sezonowości...")
    
    # Agregacja: średnia wieloletnia per kraj i per miesiąc
    seasonality = df.groupby(['geo', 'month', 'c_resid'])['OBS_VALUE'].agg(['mean', 'std']).reset_index()
    seasonality.columns = ['geo', 'month', 'c_resid', 'avg_nights', 'std_nights']
    
    # Identyfikacja miesiąca z najwyższą i najniższą liczbą noclegów
    seasonality_summary = seasonality.groupby(['geo', 'c_resid']).agg({
        'avg_nights': ['idxmax', 'idxmin', 'max', 'min']
    }).reset_index()
    
    # Dodanie nazw miesięcy dla czytelności
    month_names = {
        1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień',
        5: 'Maj', 6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień',
        9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
    }
    seasonality['month_name'] = seasonality['month'].map(month_names)
    
    print(f"Przeanalizowano sezonowość dla {seasonality['geo'].nunique()} krajów/regionów")
    
    return seasonality


def identify_top_destinations(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Identyfikuje TOP N popularnych destynacji turystycznych.
    
    Oblicza TOP 10 krajów/regionów o najwyższej liczbie noclegów,
    osobno dla turystów krajowych i zagranicznych.
    
    Args:
        df: DataFrame z danymi
        top_n: Liczba destynacji do zwrócenia (domyślnie 10)
        
    Returns:
        Słownik z TOP destynacjami dla każdego typu turysty
    """
    print(f"\nIdentyfikacja TOP {top_n} destynacji...")
    
    top_destinations = {}
    
    # Agregacja: suma noclegów per kraj
    country_totals = df.groupby(['geo', 'c_resid'])['OBS_VALUE'].sum().reset_index()
    
    # TOP dla każdego typu turysty
    for tourist_type in df['c_resid'].unique():
        type_data = country_totals[country_totals['c_resid'] == tourist_type].copy()
        type_data = type_data.sort_values('OBS_VALUE', ascending=False).head(top_n)
        top_destinations[tourist_type] = type_data
    
        print(f"  {tourist_type}: {len(type_data)} destynacji")
    
    return top_destinations


def plot_seasonality(seasonality_df: pd.DataFrame, output_dir: Path = Path("output")):
    """
    Generuje wykres liniowy sezonowości (miesiące vs liczba noclegów).
    
    Wykres pokazuje średnią wieloletnią liczbę noclegów w poszczególnych miesiącach
    dla wybranych krajów o największej sezonowości.
    
    Interpretacja:
    - Ostre szczyty w miesiącach letnich (lipiec-sierpień) wskazują na silną sezonowość turystyczną
    - Płaskie krzywe sugerują turystykę bardziej biznesową/stałą przez cały rok
    """
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerowanie wykresu sezonowości...")
    
    # Wybór krajów z największą różnicą między miesiącem max a min (najsilniejsza sezonowość)
    seasonality_wide = seasonality_df.pivot_table(
        index=['geo', 'c_resid'], 
        columns='month', 
        values='avg_nights'
    ).reset_index()
    
    month_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    available_cols = [col for col in month_cols if col in seasonality_wide.columns]
    
    max_vals = seasonality_wide[available_cols].max(axis=1)
    min_vals = seasonality_wide[available_cols].min(axis=1)
    # Zastąp 0 wartościami minimalnymi > 0, aby uniknąć dzielenia przez zero
    min_vals = min_vals.replace(0, np.nan)
    seasonality_wide['seasonality_index'] = max_vals / min_vals
    seasonality_wide['seasonality_index'] = seasonality_wide['seasonality_index'].fillna(1.0)
    
    # TOP 5 krajów z najsilniejszą sezonowością
    top_seasonal = seasonality_wide.nlargest(5, 'seasonality_index')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for idx, tourist_type in enumerate(seasonality_df['c_resid'].unique()):
        ax = axes[idx-1]
        type_data = seasonality_df[seasonality_df['c_resid'] == tourist_type]
        
        # Filtrowanie do TOP sezonowych krajów dla tego typu
        top_for_type = top_seasonal[
            top_seasonal['c_resid'] == tourist_type
        ]['geo'].head(5).tolist()
        
        if not top_for_type:
            continue
            
        for country in top_for_type:
            country_data = type_data[type_data['geo'] == country].sort_values('month')
            ax.plot(
                country_data['month'],
                country_data['avg_nights'],
                marker='o',
                label=country,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Miesiąc', fontsize=12, fontweight='bold')
        ax.set_ylabel('Średnia liczba noclegów', fontsize=12, fontweight='bold')
        ax.set_title(f'Sezonowość turystyki - {tourist_type}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze', 
                           'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru'])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'sezonowosc_turystyki.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Zapisano: {output_path}")
    plt.close()


def plot_top_destinations(top_destinations: dict, output_dir: Path = Path("output")):
    """
    Generuje wykres słupkowy TOP 10 krajów pod względem noclegów.
    
    Wykres pokazuje kraje o najwyższej łącznej liczbie noclegów,
    osobno dla turystów krajowych i zagranicznych.
    """
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerowanie wykresu TOP destynacji...")
    
    n_types = len(top_destinations)
    fig, axes = plt.subplots(1, n_types, figsize=(16, 6))
    
    if n_types == 1:
        axes = [axes]
    
    for idx, (tourist_type, data) in enumerate(top_destinations.items()):
        ax = axes[idx]
        
        bars = ax.barh(
            data['geo'].values,
            data['OBS_VALUE'].values,
            color=sns.color_palette("viridis", len(data))
        )
        
        ax.set_xlabel('Łączna liczba noclegów', fontsize=12, fontweight='bold')
        ax.set_ylabel('Kraj / Region', fontsize=12, fontweight='bold')
        ax.set_title(f'TOP 10 Destynacji - {tourist_type}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Dodanie wartości na słupkach
        for i, (bar, value) in enumerate(zip(bars, data['OBS_VALUE'].values)):
            ax.text(
                value,
                i,
                f' {value:,.0f}'.replace(',', ' '),
                va='center',
                fontsize=9
            )
    
    plt.tight_layout()
    output_path = output_dir / 'top_10_destynacji.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Zapisano: {output_path}")
    plt.close()


def plot_temporal_trends(df: pd.DataFrame, selected_country: str = None, 
                        output_dir: Path = Path("output")):
    """
    Generuje wykres liniowy zmian liczby noclegów w czasie dla wybranego kraju.
    
    Jeśli nie podano kraju, wybiera kraj z TOP 3 destynacji.
    """
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerowanie wykresu trendów czasowych...")
    
    if selected_country is None:
        # Wybór kraju z najwyższą sumą noclegów
        country_totals = df.groupby('geo')['OBS_VALUE'].sum().sort_values(ascending=False)
        selected_country = country_totals.index[0]
    
    country_data = df[df['geo'] == selected_country].copy()
    
    if len(country_data) == 0:
        print(f"  Uwaga: Brak danych dla kraju {selected_country}")
        return
    
    # Agregacja per miesiąc i typ turysty
    time_series = country_data.groupby(['TIME_PERIOD', 'c_resid'])['OBS_VALUE'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for tourist_type in time_series['c_resid'].unique():
        type_data = time_series[time_series['c_resid'] == tourist_type].sort_values('TIME_PERIOD')
        ax.plot(
            type_data['TIME_PERIOD'],
            type_data['OBS_VALUE'],
            marker='o',
            label=tourist_type,
            linewidth=2,
            markersize=4
        )
    
    ax.set_xlabel('Czas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Liczba noclegów', fontsize=12, fontweight='bold')
    ax.set_title(f'Zmiany liczby noclegów w czasie - {selected_country}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Formatowanie osi X
    fig.autofmt_xdate()
    
    plt.tight_layout()
    output_path = output_dir / f'trendy_czasowe_{selected_country.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Zapisano: {output_path}")
    plt.close()


def generate_analytical_commentary(seasonality_df: pd.DataFrame, 
                                   top_destinations: dict) -> str:
    """
    Generuje komentarz analityczny na podstawie wyników analizy.
    
    Identyfikuje:
    - Kraje z silną sezonowością
    - Kraje o bardziej biznesowym charakterze (mniejsza sezonowość)
    """
    commentary = "\n" + "="*70 + "\n"
    commentary += "KOMENTARZ ANALITYCZNY\n"
    commentary += "="*70 + "\n\n"
    
    # Analiza sezonowości
    seasonality_wide = seasonality_df.pivot_table(
        index=['geo', 'c_resid'], 
        columns='month', 
        values='avg_nights'
    ).reset_index()
    
    month_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    available_cols = [col for col in month_cols if col in seasonality_wide.columns]
    
    max_vals = seasonality_wide[available_cols].max(axis=1)
    min_vals = seasonality_wide[available_cols].min(axis=1)
    # Zastąp 0 wartościami minimalnymi > 0, aby uniknąć dzielenia przez zero
    min_vals = min_vals.replace(0, np.nan)
    seasonality_wide['seasonality_index'] = max_vals / min_vals
    seasonality_wide['seasonality_index'] = seasonality_wide['seasonality_index'].fillna(1.0)
    
    commentary += "📊 ANALIZA SEZONOWOŚCI:\n"
    commentary += "-" * 70 + "\n"
    
    # TOP 5 krajów z najsilniejszą sezonowością
    top_seasonal = seasonality_wide.nlargest(5, 'seasonality_index')
    commentary += "\nKraje o SILNEJ SEZONOWOŚCI (wysoki wskaźnik sezonowości):\n"
    for _, row in top_seasonal.iterrows():
        commentary += f"  • {row['geo']} ({row['c_resid']}): wskaźnik {row['seasonality_index']:.2f}\n"
        commentary += f"    → Silny szczyt w miesiącach letnich, prawdopodobnie turystyka wypoczynkowa\n"
    
    # Kraje o niskiej sezonowości
    low_seasonal = seasonality_wide.nsmallest(5, 'seasonality_index')
    commentary += "\nKraje o NISKIEJ SEZONOWOŚCI (stabilny ruch przez cały rok):\n"
    for _, row in low_seasonal.iterrows():
        commentary += f"  • {row['geo']} ({row['c_resid']}): wskaźnik {row['seasonality_index']:.2f}\n"
        commentary += f"    → Bardziej równomierny rozkład, prawdopodobnie turystyka biznesowa/stabilna\n"
    
    # TOP destynacje
    commentary += "\n\n📈 TOP DESTYNACJE TURYSTYCZNE:\n"
    commentary += "-" * 70 + "\n"
    for tourist_type, data in top_destinations.items():
        commentary += f"\n{tourist_type}:\n"
        for idx, row in data.head(5).iterrows():
            commentary += f"  {idx + 1}. {row['geo']}: {row['OBS_VALUE']:,.0f} noclegów\n"
    
    commentary += "\n" + "="*70 + "\n"
    
    return commentary


def main():
    """
    Główna funkcja aplikacji - wykonuje pełną analizę turystyczną.
    """
    print("="*70)
    print("ANALIZA TURYSTYCZNA - ETAP 2")
    print("Wyszukiwarka popularnych miejsc turystycznych")
    print("="*70)
    
    # Ścieżka do danych
    csv_path = "data/tour_occ_nim__custom_15171914_linear.csv"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Wczytanie danych
    df = load_data(csv_path)
    
    # 2. Czyszczenie danych
    df_clean = clean_data(df)
    
    # 3. Analiza sezonowości
    seasonality = analyze_seasonality(df_clean)
    
    # 4. Identyfikacja TOP destynacji
    top_destinations = identify_top_destinations(df_clean, top_n=10)
    
    # 5. Generowanie wykresów
    plot_seasonality(seasonality, output_dir)
    plot_top_destinations(top_destinations, output_dir)
    plot_temporal_trends(df_clean, selected_country=None, output_dir=output_dir)
    
    # 6. Komentarz analityczny
    commentary = generate_analytical_commentary(seasonality, top_destinations)
    print(commentary)
    
    # Zapisanie komentarza do pliku
    commentary_path = output_dir / "komentarz_analityczny.txt"
    with open(commentary_path, 'w', encoding='utf-8') as f:
        f.write(commentary)
    print(f"Zapisano komentarz analityczny: {commentary_path}")
    
    # Zapisanie wyników do plików CSV
    seasonality.to_csv(output_dir / "sezonowosc.csv", index=False, encoding='utf-8')
    print(f"\nZapisano wyniki analizy sezonowości: {output_dir / 'sezonowosc.csv'}")
    
    print("\n" + "="*70)
    print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print(f"Wszystkie wyniki zapisane w katalogu: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

