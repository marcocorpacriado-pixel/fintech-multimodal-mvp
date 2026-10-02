import sys
import os
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from edgar import *

# --- 1. Forzar codificación UTF-8 en Windows para consola ---
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# --- 2. Logging con encoding UTF-8 explícito ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_xbrl.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Identificación ante la SEC ---
set_identity("Marco Corpa marcocorpacriado@gmail.com")

# --- Directorio de salida ---
XBRL_DIR = Path("./data/raw/xbrl")
XBRL_DIR.mkdir(parents=True, exist_ok=True)

# --- Tickers ---
tickers = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "JPM", "BAC", "GS",
    "JNJ", "UNH",
    "AMZN", "TSLA"
]

def sanitize_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia tipos de datos y nulos para evitar errores de PyArrow en to_parquet.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Nombres de columna a string puro
    df.columns = [str(col) for col in df.columns]

    # Reemplazar cadenas vacías o espacios en blanco por np.nan
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace({"-": np.nan, "—": np.nan})

    # Convertir columnas numéricas que quedaron como 'object' por culpa de los strings vacíos
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # Si es columna textual (concept, label, description), se conserva
            pass

    return df

def extract_statements(xbrl):
    """
    Extrae los estados financieros principales de un objeto XBRL.
    """
    statements = {}
    stmts = getattr(xbrl, "statements", None)
    if stmts is None:
        return statements

    target_statements = {
        "balance_sheet": "balance_sheet",
        "income_statement": "income_statement",
        "cash_flow_statement": "cash_flow_statement",
        "statement_of_equity": "statement_of_equity",
        "comprehensive_income": "comprehensive_income",
    }

    for name, attr in target_statements.items():
        try:
            stmt = getattr(stmts, attr, None)
            if callable(stmt):
                stmt = stmt()

            if stmt is not None:
                df = stmt.to_dataframe()
                if df is not None and not df.empty:
                    statements[name] = sanitize_dataframe_for_parquet(df)
        except Exception as e:
            logging.warning(f"    ⚠️ No se pudo extraer '{name}': {type(e).__name__}: {e}")

    return statements

def main():
    logging.info("=" * 60)
    logging.info("INICIANDO EXTRACCIÓN DE ESTADOS FINANCIEROS (XBRL)")
    logging.info("=" * 60)

    total_ok = 0
    total_skip = 0
    total_error = 0

    for ticker in tickers:
        logging.info(f"\n📊 Procesando {ticker}...")
        try:
            company = Company(ticker)
            filings = company.get_filings(
                form=["10-K", "10-Q", "10-K/A", "10-Q/A"],
                filing_date="2024-01-01:"
            )

            if not filings:
                logging.warning(f"  ⚠️ No se encontraron filings para {ticker}")
                continue

            for i, filing in enumerate(filings, 1):
                safe_form = filing.form.replace("/", "_")
                accession = getattr(filing, "accession_number", getattr(filing, "accession_no", "unknown"))
                base_name = f"{ticker}_{safe_form}_{filing.filing_date}_{accession}"

                logging.info(f"  [{i}/{len(filings)}] {filing.form} - {filing.filing_date}")

                try:
                    xbrl = filing.xbrl()
                    if xbrl is None:
                        logging.warning("    ⚠️ Sin XBRL (filing sin datos estructurados)")
                        total_skip += 1
                        continue

                    # --- Extraer estados financieros ---
                    statements = extract_statements(xbrl)

                    if not statements:
                        logging.warning("    ⚠️ No se extrajo ningún estado financiero")
                        total_skip += 1
                        continue

                    # --- Guardar cada estado en Parquet ---
                    for stmt_name, df in statements.items():
                        parquet_path = XBRL_DIR / f"{base_name}__{stmt_name}.parquet"
                        df.to_parquet(parquet_path, index=False)

                    logging.info(f"    ✅ {len(statements)} estados guardados: {list(statements.keys())}")
                    total_ok += 1

                except Exception as e:
                    logging.error(f"    ❌ Error extrayendo XBRL: {type(e).__name__}: {e}")
                    total_error += 1

                time.sleep(0.3)

        except Exception as e:
            logging.error(f"❌ Error crítico con {ticker}: {type(e).__name__}: {e}")
            time.sleep(1)

    logging.info("\n" + "=" * 60)
    logging.info("✅ EXTRACCIÓN COMPLETADA")
    logging.info(f"   Filings procesados con éxito: {total_ok}")
    logging.info(f"   Omitidos (sin XBRL o sin estados): {total_skip}")
    logging.info(f"   Errores: {total_error}")
    logging.info(f"   Directorio: {XBRL_DIR}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()