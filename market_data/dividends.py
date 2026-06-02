from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


_ZB_CODES: Dict[str, str] = {
    "AC":    "ACCOR-4601",
    "AI":    "AIR-LIQUIDE-4605",
    "AIR":   "AIRBUS-4637",
    "MT":    "ARCELORMITTAL-34942237",
    "CS":    "AXA-4670",
    "BNP":   "BNP-PARIBAS-4618",
    "EN":    "BOUYGUES-4995",
    "CAP":   "CAPGEMINI-4624",
    "CA":    "CARREFOUR-4626",
    "ACA":   "CREDIT-AGRICOLE-S-A-4735",
    "BN":    "DANONE-4634",
    "DSY":   "DASSAULT-SYSTEMES-SE-4635",
    "FGR":   "EIFFAGE-S-A-4638",
    "ENGI":  "ENGIE-4995",
    "EL":    "ESSILORLUXOTTICA-4641",
    "ERF":   "EUROFINS-SCIENTIFIC-SE-4753",
    "ENX":   "EEURONEXT-N-V-16725768",
    "RMS":   "HERMES-INTERNATIONAL-4657",
    "KER":   "KERING-4683",
    "LR":    "LEGRAND-16719",
    "OR":    "L-OREAL-4666",
    "MC":    "LVMH-4669",
    "ML":    "MICHELIN-4672",
    "ORA":   "ORANGE-4649",
    "RI":    "PERNOD-RICARD-4681",
    "PUB":   "PUBLICIS-GROUPE-S-A-4685",
    "RNO":   "RENAULT-4688",
    "SAF":   "SAFRAN-4696",
    "SGO":   "SAINT-GOBAIN-4697",
    "SAN":   "SANOFI-4698",
    "SU":    "SCHNEIDER-ELECTRIC-4699",
    "GLE":   "SOCIETE-GENERALE-4702",
    "STLA":  "STELLANTIS-N-V-117814143",
    "STM":   "STMICROELECTRONICS-N-V-4710",
    "HO":    "THALES-4715",
    "TTE":   "TOTALENERGIES-SE-4717",
    "URW":   "UNIBAIL-RODAMCO-WESTFIELD-43851519",
    "VIE":   "VEOLIA-ENVIRONNEMENT-4726",
    "DG":    "VINCI-4725",
    "BVI":   "BUREAU-VERITAS-SA-64670",
}


@dataclass(frozen=True)
class DividendYield:
    ticker:         str
    last_yield:     float
    next_yield_est: float
    avg_3y_yield:   float
    all_yields:     Dict[str, float] = field(default_factory=dict)

    @property
    def recommended(self) -> float:
        return self.avg_3y_yield


class DividendCurve:
    def __init__(self, ticker: str, continuous_yield: float) -> None:
        self.ticker = ticker.upper()
        self._q     = continuous_yield

    @property
    def q(self) -> float:
        return self._q

    @classmethod
    def from_dividend_yield(cls, dy: DividendYield) -> "DividendCurve":
        return cls(dy.ticker, dy.recommended)

    def __repr__(self) -> str:
        return f"DividendCurve(ticker={self.ticker!r}, q={self._q:.4%})"


def _parse_table_row(
    cells: list,
    years: list[str],
    divisor: float = 1.0,
) -> Dict[str, Dict]:
    """Parse a data row: returns {year: {value, estimate}} with None for dashes."""
    result: Dict[str, Dict] = {}
    for i, year in enumerate(years):
        ci = i + 1
        if ci >= len(cells):
            break
        cell = cells[ci]
        is_est = "table-child--bg-estimates" in " ".join(cell.get("class", []))
        raw = (
            cell.get_text()
            .strip()
            .replace("\xa0", "")
            .replace("%", "")
            .replace(" ", "")
            .replace(" ", "")
            .replace(",", ".")
            .strip()
        )
        try:
            val: Optional[float] = float(raw) / divisor
        except ValueError:
            val = None
        result[year] = {"value": val, "estimate": is_est}
    return result


def _extract_current_price(soup: BeautifulSoup) -> Optional[float]:
    """Try to extract the current stock price from a zonebourse page."""
    candidates = [
        soup.find(itemprop="price"),
        soup.select_one("[data-type='last']"),
        soup.select_one(".c-faceplate__price"),
        soup.select_one(".cotation-cours"),
        soup.select_one("#cotation-cours"),
        soup.select_one("span.cotation"),
    ]
    for el in candidates:
        if el is None:
            continue
        text = el.get("content") or el.get_text()
        raw = re.sub(r"[^\d,.]", "", text).replace(",", ".").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def fetch_dividend(ticker: str) -> DividendYield:
    code = _ZB_CODES.get(ticker.upper())
    if not code:
        raise ValueError(
            f"Ticker {ticker!r} inconnu. Disponibles : {', '.join(sorted(_ZB_CODES))}"
        )

    url = f"https://www.zonebourse.com/cours/action/{code}/valorisation-dividende/"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="fr-FR",
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        for table_id in ("dividendTable", "valuationEnterpriseTable"):
            try:
                page.wait_for_selector(f"table#{table_id}", timeout=10000)
                break
            except Exception:
                continue
        # Attendre que le JS ait rempli au moins une cellule non-tiret dans le tableau
        try:
            page.wait_for_function(
                """() => {
                    const rows = document.querySelectorAll('#dividendTable tbody tr, #valuationEnterpriseTable tbody tr');
                    for (const row of rows) {
                        const cells = row.querySelectorAll('td');
                        for (let i = 1; i < cells.length; i++) {
                            const t = cells[i].textContent.trim();
                            if (t && t !== '-') return true;
                        }
                    }
                    return false;
                }""",
                timeout=12000,
            )
        except Exception:
            page.wait_for_timeout(5000)
        html = page.content()
        browser.close()

    soup  = BeautifulSoup(html, "html.parser")
    table = (
        soup.find("table", id="dividendTable")
        or soup.find("table", id="valuationEnterpriseTable")
    )
    if not table:
        raise RuntimeError(f"Table dividende introuvable pour {ticker}")

    years = [
        th.get_text().strip()
        for th in table.select("thead th")
        if re.match(r"^\d{4}$", th.get_text().strip())
    ]

    yield_row:   Dict[str, Dict] = {}
    div_ps_row:  Dict[str, Dict] = {}

    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        label = cells[0].get_text().lower()
        if "taux de rendement" in label or "rendement" in label:
            yield_row  = _parse_table_row(cells, years, divisor=100.0)
        elif "dividende" in label and "action" in label:
            div_ps_row = _parse_table_row(cells, years, divisor=1.0)

    # Prefer explicit yield% row; fall back to div/share ÷ current price
    yields: Dict[str, Dict] = {}
    if yield_row and any(d["value"] for d in yield_row.values() if d["value"]):
        yields = yield_row
    elif div_ps_row:
        price = _extract_current_price(soup)
        if price:
            yields = {
                y: {"value": d["value"] / price if d["value"] is not None else None,
                    "estimate": d["estimate"]}
                for y, d in div_ps_row.items()
            }

    def _val(d: Dict) -> float:
        return d["value"] if d["value"] is not None else 0.0

    historical = {y: d for y, d in yields.items() if not d["estimate"]}
    estimated  = {y: d for y, d in yields.items() if d["estimate"]}

    last_hist  = _val(yields[max(historical)]) if historical else 0.0
    next_est   = _val(yields[min(estimated)])  if estimated  else 0.0

    hist_vals = sorted(
        [_val(d) for d in historical.values()], reverse=True
    )[:3]
    avg_3y = sum(hist_vals) / len(hist_vals) if hist_vals else 0.0

    return DividendYield(
        ticker         = ticker.upper(),
        last_yield     = last_hist,
        next_yield_est = next_est,
        avg_3y_yield   = round(avg_3y, 5),
        all_yields     = {y: _val(d) for y, d in yields.items()},
    )


def flat_dividend(ticker: str, q: float) -> DividendCurve:
    return DividendCurve(ticker, q)
