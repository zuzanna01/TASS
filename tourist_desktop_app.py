"""
Aplikacja desktopowa - Wyszukiwarka Popularnych Miejsc Turystycznych
Interaktywna aplikacja z GUI do wyboru kraju i wyświetlania wyników analizy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from pathlib import Path
import warnings
import sys

# Import funkcji z głównego modułu analizy
from tourist_analysis import load_data, clean_data, analyze_seasonality

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


class TouristAnalysisApp:
    """Główna klasa aplikacji desktopowej"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wyszukiwarka Popularnych Miejsc Turystycznych")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Zmienne do przechowywania danych
        self.df_clean = None
        self.seasonality_df = None
        self.countries_list = []
        
        # Słownik tłumaczeń typów turystyki
        self.tourist_type_translation = {
            'Domestic country': 'Turystyka krajowa',
            'Foreign country': 'Turystyka zagraniczna',
            'Total': 'Ogółem'
        }
        
        # Tworzenie interfejsu
        self.create_widgets()
        
        # Wczytanie danych przy starcie
        self.load_data_on_startup()
    
    def translate_tourist_type(self, tourist_type: str) -> str:
        """Tłumaczy typ turystyki na polski"""
        return self.tourist_type_translation.get(tourist_type, tourist_type)
    
    def create_widgets(self):
        """Tworzy elementy interfejsu użytkownika"""
        
        # Główny kontener
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfiguracja siatki
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Panel wyboru kraju
        selection_frame = ttk.LabelFrame(main_frame, text="Wybór kraju", padding="10")
        selection_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(selection_frame, text="Wybierz kraj/region:").grid(row=0, column=0, padx=5, pady=5)
        
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.country_var,
            state="readonly",
            width=40
        )
        self.country_combo.grid(row=0, column=1, padx=5, pady=5)
        self.country_combo.bind("<<ComboboxSelected>>", self.on_country_selected)
        
        ttk.Button(
            selection_frame,
            text="Załaduj dane",
            command=self.load_data_on_startup
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Panel statystyk
        stats_frame = ttk.LabelFrame(main_frame, text="Statystyki", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        # Kontener dla tekstu i scrollbara
        text_container = ttk.Frame(stats_frame)
        text_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_container.columnconfigure(0, weight=1)
        text_container.rowconfigure(0, weight=1)
        
        self.stats_text = tk.Text(text_container, width=45, wrap=tk.WORD, 
                                  font=('Consolas', 10), 
                                  bg='#ffffff', fg='#000000',
                                  padx=10, pady=10,
                                  relief=tk.FLAT, borderwidth=1)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_stats = ttk.Scrollbar(text_container, command=self.stats_text.yview)
        scrollbar_stats.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.stats_text.config(yscrollcommand=scrollbar_stats.set)
        
        # Panel wykresów
        charts_frame = ttk.LabelFrame(main_frame, text="Wykresy", padding="10")
        charts_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Notebook dla różnych wykresów
        self.notebook = ttk.Notebook(charts_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Zakładki dla wykresów
        self.seasonality_frame = ttk.Frame(self.notebook)
        self.temporal_frame = ttk.Frame(self.notebook)
        self.comparison_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.seasonality_frame, text="Sezonowość")
        self.notebook.add(self.temporal_frame, text="Trendy czasowe")
        self.notebook.add(self.comparison_frame, text="Porównanie")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Gotowy")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_data_on_startup(self):
        """Wczytuje dane przy starcie aplikacji"""
        self.status_var.set("Wczytywanie danych...")
        self.root.update()
        
        try:
            csv_path = "data/tour_occ_nim__custom_15171914_linear.csv"
            
            if not Path(csv_path).exists():
                messagebox.showerror("Błąd", f"Nie znaleziono pliku: {csv_path}")
                self.status_var.set("Błąd wczytywania danych")
                return
            
            # Wczytanie i czyszczenie danych
            df = load_data(csv_path)
            self.df_clean = clean_data(df)
            
            # Analiza sezonowości
            self.seasonality_df = analyze_seasonality(self.df_clean)
            
            # Pobranie listy krajów
            self.countries_list = sorted(self.df_clean['geo'].unique().tolist())
            self.country_combo['values'] = self.countries_list
            
            self.status_var.set(f"Wczytano dane dla {len(self.countries_list)} krajów/regionów")
            
            # Automatyczny wybór pierwszego kraju jeśli dostępny
            if self.countries_list:
                self.country_var.set(self.countries_list[0])
                self.on_country_selected()
        
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas wczytywania danych:\n{str(e)}")
            self.status_var.set("Błąd wczytywania danych")
    
    def on_country_selected(self, event=None):
        """Obsługuje wybór kraju z listy"""
        selected_country = self.country_var.get()
        
        if not selected_country or self.df_clean is None:
            return
        
        self.status_var.set(f"Analizowanie danych dla: {selected_country}")
        self.root.update()
        
        try:
            # Wyświetlenie statystyk
            self.display_statistics(selected_country)
            
            # Wyświetlenie wykresów
            self.display_seasonality_chart(selected_country)
            self.display_temporal_chart(selected_country)
            self.display_comparison_chart(selected_country)
            
            self.status_var.set(f"Wyświetlono wyniki dla: {selected_country}")
        
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas analizy:\n{str(e)}")
            self.status_var.set("Błąd analizy")
    
    def display_statistics(self, country: str):
        """Wyświetla statystyki dla wybranego kraju"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        country_data = self.df_clean[self.df_clean['geo'] == country]
        seasonality_country = self.seasonality_df[self.seasonality_df['geo'] == country]
        
        month_names_full = {
            1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień',
            5: 'Maj', 6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień',
            9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
        }
        
        month_names_short = {
            1: 'Sty', 2: 'Lut', 3: 'Mar', 4: 'Kwi',
            5: 'Maj', 6: 'Cze', 7: 'Lip', 8: 'Sie',
            9: 'Wrz', 10: 'Paź', 11: 'Lis', 12: 'Gru'
        }
        
        stats_text = "=" * 55 + "\n"
        stats_text += f"KRAJ/REGION: {country.upper()}\n"
        stats_text += "=" * 55 + "\n\n"
        
        # Statystyki ogólne
        total_nights = country_data['OBS_VALUE'].sum()
        avg_per_month = country_data.groupby(['TIME_PERIOD', 'c_resid'])['OBS_VALUE'].sum().mean()
        years_available = sorted(country_data['TIME_PERIOD'].dt.year.unique())
        
        stats_text += "STATYSTYKI OGOLNE\n"
        stats_text += "-" * 55 + "\n"
        stats_text += f"Laczna liczba noclegow: {total_nights:,.0f}\n"
        stats_text += f"Srednia miesieczna:     {avg_per_month:,.0f}\n"
        stats_text += f"Zakres danych:          {min(years_available)} - {max(years_available)}\n"
        stats_text += f"Liczba lat:             {len(years_available)}\n\n"
        
        # Analiza per typ turysty
        for tourist_type in sorted(country_data['c_resid'].unique()):
            type_data = country_data[country_data['c_resid'] == tourist_type]
            type_seasonality = seasonality_country[seasonality_country['c_resid'] == tourist_type]
            translated_type = self.translate_tourist_type(tourist_type)
            
            stats_text += "\n" + "=" * 55 + "\n"
            stats_text += f"{translated_type.upper()}\n"
            stats_text += "-" * 55 + "\n"
            
            total_type = type_data['OBS_VALUE'].sum()
            avg_type = type_data['OBS_VALUE'].mean()
            max_type = type_data['OBS_VALUE'].max()
            min_type = type_data['OBS_VALUE'].min()
            
            stats_text += f"Laczna liczba noclegow:  {total_type:,.0f}\n"
            stats_text += f"Srednia liczba noclegow: {avg_type:,.0f}\n"
            stats_text += f"Maksimum:                {max_type:,.0f}\n"
            stats_text += f"Minimum:                 {min_type:,.0f}\n"
            
            if len(type_seasonality) > 0:
                # Miesiąc z najwyższą liczbą noclegów
                max_month_data = type_seasonality.loc[type_seasonality['avg_nights'].idxmax()]
                min_month_data = type_seasonality.loc[type_seasonality['avg_nights'].idxmin()]
                
                stats_text += "\nANALIZA SEZONOWOSCI\n"
                stats_text += "-" * 55 + "\n"
                stats_text += f"Najwyzszy sezon:\n"
                stats_text += f"  Miesiac: {month_names_full[max_month_data['month']]}\n"
                stats_text += f"  Liczba noclegow: {max_month_data['avg_nights']:,.0f}\n"
                stats_text += f"  Odchylenie std:  {max_month_data['std_nights']:,.0f}\n\n"
                
                stats_text += f"Najnizszy sezon:\n"
                stats_text += f"  Miesiac: {month_names_full[min_month_data['month']]}\n"
                stats_text += f"  Liczba noclegow: {min_month_data['avg_nights']:,.0f}\n"
                stats_text += f"  Odchylenie std:  {min_month_data['std_nights']:,.0f}\n\n"
                
                # Wskaźnik sezonowości
                seasonality_index = max_month_data['avg_nights'] / min_month_data['avg_nights'] if min_month_data['avg_nights'] > 0 else 0
                stats_text += f"Wskaznik sezonowosci: {seasonality_index:.2f}\n"
                
                if seasonality_index > 3:
                    stats_text += "Charakter: SILNA SEZONOWOSC\n"
                    stats_text += "Typ turystyki: Wypoczynkowa\n"
                elif seasonality_index < 2:
                    stats_text += "Charakter: NISKA SEZONOWOSC\n"
                    stats_text += "Typ turystyki: Biznesowa/Stala\n"
                else:
                    stats_text += "Charakter: UMIARKOWANA SEZONOWOSC\n"
                    stats_text += "Typ turystyki: Mieszana\n"
        
        # TOP 5 miesięcy
        stats_text += "\n" + "=" * 55 + "\n"
        stats_text += "TOP 5 MIESIECY (srednia wieloletnia)\n"
        stats_text += "-" * 55 + "\n"
        top_months = seasonality_country.nlargest(5, 'avg_nights')
        for rank, (idx, row) in enumerate(top_months.iterrows(), 1):
            month_name = month_names_short.get(row['month'], str(row['month']))
            translated_type = self.translate_tourist_type(row['c_resid'])
            stats_text += f"{rank}. {month_name:3} | {row['avg_nights']:>12,.0f} | {translated_type}\n"
        
        stats_text += "\n" + "=" * 55 + "\n"
        
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)
    
    def clear_frame(self, frame):
        """Czyści wszystkie widgety z ramki"""
        for widget in frame.winfo_children():
            widget.destroy()
    
    def display_seasonality_chart(self, country: str):
        """Wyświetla wykres sezonowości dla wybranego kraju"""
        self.clear_frame(self.seasonality_frame)
        
        country_seasonality = self.seasonality_df[self.seasonality_df['geo'] == country]
        
        if len(country_seasonality) == 0:
            label = ttk.Label(self.seasonality_frame, text="Brak danych dla tego kraju")
            label.pack(expand=True)
            return
        
        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=(10, 6))
        
        month_names = ['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze',
                      'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru']
        
        for tourist_type in country_seasonality['c_resid'].unique():
            type_data = country_seasonality[country_seasonality['c_resid'] == tourist_type].sort_values('month')
            translated_label = self.translate_tourist_type(tourist_type)
            ax.plot(
                type_data['month'],
                type_data['avg_nights'],
                marker='o',
                label=translated_label,
                linewidth=2.5,
                markersize=8
            )
        
        ax.set_xlabel('Miesiąc', fontsize=12, fontweight='bold')
        ax.set_ylabel('Średnia liczba noclegów', fontsize=12, fontweight='bold')
        ax.set_title(f'Sezonowość turystyki - {country}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Umieszczenie wykresu w aplikacji
        canvas = FigureCanvasTkAgg(fig, self.seasonality_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar do nawigacji
        toolbar = NavigationToolbar2Tk(canvas, self.seasonality_frame)
        toolbar.update()
    
    def display_temporal_chart(self, country: str):
        """Wyświetla wykres trendów czasowych dla wybranego kraju"""
        self.clear_frame(self.temporal_frame)
        
        country_data = self.df_clean[self.df_clean['geo'] == country]
        
        if len(country_data) == 0:
            label = ttk.Label(self.temporal_frame, text="Brak danych dla tego kraju")
            label.pack(expand=True)
            return
        
        # Agregacja per miesiąc i typ turysty
        time_series = country_data.groupby(['TIME_PERIOD', 'c_resid'])['OBS_VALUE'].sum().reset_index()
        
        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for tourist_type in time_series['c_resid'].unique():
            type_data = time_series[time_series['c_resid'] == tourist_type].sort_values('TIME_PERIOD')
            translated_label = self.translate_tourist_type(tourist_type)
            ax.plot(
                type_data['TIME_PERIOD'],
                type_data['OBS_VALUE'],
                marker='o',
                label=translated_label,
                linewidth=2,
                markersize=4
            )
        
        ax.set_xlabel('Czas', fontsize=12, fontweight='bold')
        ax.set_ylabel('Liczba noclegów', fontsize=12, fontweight='bold')
        ax.set_title(f'Zmiany liczby noclegów w czasie - {country}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        
        # Umieszczenie wykresu w aplikacji
        canvas = FigureCanvasTkAgg(fig, self.temporal_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar do nawigacji
        toolbar = NavigationToolbar2Tk(canvas, self.temporal_frame)
        toolbar.update()
    
    def display_comparison_chart(self, country: str):
        """Wyświetla wykres porównawczy (sezonowość vs średnia)"""
        self.clear_frame(self.comparison_frame)
        
        country_seasonality = self.seasonality_df[self.seasonality_df['geo'] == country]
        country_data = self.df_clean[self.df_clean['geo'] == country]
        
        if len(country_seasonality) == 0:
            label = ttk.Label(self.comparison_frame, text="Brak danych dla tego kraju")
            label.pack(expand=True)
            return
        
        # Obliczenie średniej dla każdego typu turysty
        avg_by_type = country_data.groupby('c_resid')['OBS_VALUE'].mean()
        
        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=(10, 6))
        
        month_names = ['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze',
                      'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru']
        
        for tourist_type in country_seasonality['c_resid'].unique():
            type_data = country_seasonality[country_seasonality['c_resid'] == tourist_type].sort_values('month')
            avg_value = avg_by_type.get(tourist_type, 0)
            translated_label = self.translate_tourist_type(tourist_type)
            
            ax.plot(
                type_data['month'],
                type_data['avg_nights'],
                marker='o',
                label=f'{translated_label} - rzeczywiste',
                linewidth=2.5,
                markersize=8
            )
            
            # Linia średniej
            ax.axhline(
                y=avg_value,
                color=ax.lines[-1].get_color(),
                linestyle='--',
                alpha=0.5,
                label=f'{translated_label} - średnia ({avg_value:,.0f})'
            )
        
        ax.set_xlabel('Miesiąc', fontsize=12, fontweight='bold')
        ax.set_ylabel('Liczba noclegów', fontsize=12, fontweight='bold')
        ax.set_title(f'Sezonowość vs Średnia - {country}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Umieszczenie wykresu w aplikacji
        canvas = FigureCanvasTkAgg(fig, self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar do nawigacji
        toolbar = NavigationToolbar2Tk(canvas, self.comparison_frame)
        toolbar.update()


def main():
    """Uruchamia aplikację desktopową"""
    root = tk.Tk()
    app = TouristAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

