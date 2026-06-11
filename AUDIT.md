# Audit du projet `options-pricing-lib`

Revue statique de l'ensemble du code (~5 600 lignes Python, 21 fichiers). Les insuffisances sont classées par sévérité. Pour chaque point : le fichier/ligne concerné et l'impact.

---

## 1. Bloquants — le projet ne tourne pas en l'état

### 1.1 Dépendance externe manquante : `Produit_pricing.pdts`
`pricers/analytical.py` (l. 13-41) et `pricers/mc.py` (l. 353-359) importent **tout le moteur de pricing analytique** (`_gbs`, `warrant`, `bonus_certificate`, `_rc`, `implied_participation_*`, `_barrier`, etc.) depuis un package `Produit_pricing` **qui n'est pas dans le dépôt**. Le code remonte de trois niveaux dans le système de fichiers (`sys.path.insert`) pour aller le chercher ailleurs sur la machine d'origine.

Conséquence : la « version publique » est **non fonctionnelle de façon autonome**. Tous les prix analytiques (vanilles, certificats catégorie 1/2/3, RC) dépendent d'un code absent. C'est l'insuffisance la plus grave.

### 1.2 Imports incohérents / packaging cassé
Trois conventions d'import cohabitent et se contredisent :

- `api.py` : `from market_data...`, `from pricers...` (top-level) **mais aussi** `from pricing_lib.risk import ...` (l. 19) ;
- `pricers/analytical.py`, `pricers/mc.py`, `pricers/autocalls.py`, `models/heston.py` : `from pricing_lib.market_data...`, `from pricing_lib.models...` ;
- les sous-paquets entre eux : imports relatifs (`from .rates import ...`).

Or le dossier racine s'appelle `options-pricing-lib` (nom **non importable** : il contient un tiret) et il n'existe **aucun** paquet `pricing_lib`. On compte ~20 références à `pricing_lib`. **Il n'existe aucune disposition de répertoires** sous laquelle `api.py` (qui veut `market_data` en top-level) et `analytical.py` (qui veut `pricing_lib.market_data`) s'importent tous les deux : `ImportError` garanti. Vérifié : `import api` échoue dès la première ligne.

### 1.3 Aucune déclaration de dépendances
Pas de `requirements.txt`, `pyproject.toml` ni `setup.py`. Les dépendances réelles, lourdes et implicites : `numpy`, `scipy`, `pandas`, `requests`, `beautifulsoup4`, `yfinance`, `python-dateutil`, `matplotlib`, et surtout **`playwright` + un navigateur Chrome** (`market_data/dividends.py`). Installation impossible sans deviner la liste, et Playwright exige en plus `playwright install chromium`.

---

## 2. Majeurs — exactitude financière / risque

### 2.1 Rho faux : le choc de taux n'est pas actualisé
`risk/greeks.py` : `_BumpedMarket` ne décale `r` que via sa propriété `r` (utilisée pour le forward Black-Scholes). Mais toute l'actualisation passe par `ois_curve.df_tau(...)`, et `ois_curve` est proxifié **inchangé** vers la base. Donc pour les Reverse Convertibles, les autocalls, et toute jambe actualisée par la courbe OIS, le choc de taux **n'affecte pas le discount factor**. Le rho calculé est incohérent (capture seulement l'effet sur le drift, pas sur l'actualisation).

### 2.2 Theta toujours `NaN` via l'API publique
`api.Greeks(...)` construit `FiniteDiffGreeks(_ClosurePricer(fn), None, snap)` avec `product=None`. Or `FiniteDiffGreeks.theta()` fait `hasattr(prod, "maturity")` sur `None` → `False` → retourne `float("nan")`. Le bump de maturité (`_bump_product_maturity`) n'est donc **jamais** exercé par le chemin public : theta est systématiquement NaN.

### 2.3 Méthode `sigma` inexistante dans le bump de vol
`risk/greeks.py` : `_BumpedVolSurface.sigma()` appelle `self._base.sigma(...)`, et `_BumpedMarket.sigma()` appelle `self.vol_surface.sigma(...)`. Mais `VolSurface` n'expose que `vol()`, pas `sigma()`. Ce chemin lève `AttributeError` s'il est emprunté. De plus, le vega des exotiques passe par `dupire_local_vol(surf, ...)` qui appelle en interne `surf.atm_vol()` → `VolSurface.vol()` **du sous-jacent non bumpé** : le choc de vol n'est que partiellement propagé.

### 2.4 Greeks Monte Carlo des barrières non cohérents avec le prix
Pour `bonus_certificate` et `twin_win_certificate` (`pricers/mc.py`), les Greeks sont calculés sur un payoff **sans la barrière** et avec `n_paths // 4`. Le commentaire le reconnaît (« approximation »), mais le résultat est renvoyé comme delta/gamma/vega officiels, donc incohérent avec le prix affiché (lui calculé avec barrière continue).

### 2.5 Double comptage probable du nominal dans la RC Monte Carlo
`pricers/mc.py`, `reverse_convertible` : `price = (bond_pv + redemption - nominal)/nominal*100 + 100`. Or `_bond_value` inclut déjà le remboursement du nominal actualisé (`nominal * df_tau(T)`), et `redemption` est l'espérance actualisée d'un payoff terminal qui vaut lui aussi `nominal` (ou `ST*ratio`). Le nominal semble compté **deux fois**. Formule à revalider numériquement (et à confronter au mode analytique).

### 2.6 Fallback silencieux sur les valeurs de base (`x or self.y`)
`pricers/mc.py` : motifs `S=S_override or self.S`, `r=r_override or self.r`. Si la valeur bumpée vaut `0.0` (taux nul — réaliste en zone euro — ou spot nul), Python retombe **silencieusement** sur la valeur de base. Bug latent qui fausse les sensibilités près de zéro. Utiliser `... if x is not None else ...`.

### 2.7 Calibration Heston : contrainte de Feller non imposée
`models/heston.py`, `calibrate_heston` : bornes `kappa∈[0.1,15]`, `theta∈[0.01,1]`, `sigma_v∈[0.01,2]`. Rien ne garantit `2κθ > σ_v²` → variance qui peut coller à zéro, schéma d'Euler « full truncation » biaisé. Par ailleurs `rho∈[-0.95, 0]` **interdit toute corrélation positive** (choix non documenté), et `heston_price` plafonne le résultat par `max(call, intrinsèque)` ce qui **masque** les erreurs d'intégration au lieu de les corriger. Grille `u` fixe `[1e-6, 200]` / 128 points, non adaptée aux très courtes maturités ni aux ailes profondes.

### 2.8 Un taux plat 1Y pour toutes les maturités côté BS
`market_data/market_snapshot.py` : `r = ois_curve.zero_rate_tau(1.0)`. Ce taux 1 an unique est utilisé comme taux sans risque **plat** dans toutes les formules BS, quelle que soit la maturité, alors qu'une courbe complète est disponible et utilisée, elle, pour l'actualisation (`df_tau`). Incohérence interne entre le drift et le discount.

---

## 3. Modérés — qualité, maintenabilité, robustesse

### 3.1 Aucun test
Zéro fichier de test sur une bibliothèque de pricing. Aucune vérification de parité call-put, de convergence MC, de cohérence analytique vs MC, ni de bornes d'arbitrage. Indispensable pour ce type de code.

### 3.2 Aucune documentation projet
Pas de `README`, pas de guide d'installation/usage, pas de `LICENSE` (problématique pour un dépôt présenté comme « public »).

### 3.3 Code mort et doublons
- `models/heston2.py` (`heston_monte_carlo`) est importé dans `pricers/mc.py` (l. 19) mais **jamais utilisé** — c'est une seconde implémentation Heston (Euler arithmétique, `np.random.seed` global hérité) redondante.
- `market_data/sabr_smoother.py` n'est **jamais branché** dans `VolSurface` (qui utilise `RectBivariateSpline` directement).
- `build_vol_surface.py` **redéfinit** `build_surface` et `compute_dupire_surface` déjà présents dans `vol_surface.py`, avec une logique d'échéance différente (18 du mois vs 3ᵉ vendredi) → surfaces incohérentes selon le point d'entrée.
- `dupire_example.py` réimplémente encore une **3ᵉ** variante de Dupire.

### 3.4 Bug dans `sabr_smoother.py`
Dans `_fit_sabr_to_tenor.residual`, le déballage est `alpha, beta, nu, rho = p` alors que `x0`, les `bounds` et la sortie sont ordonnés `[alpha, beta, rho, nu]` → **rho et nu sont permutés** dans la fonction objectif. De plus `sabr_vol` renvoie `sigma * np.sqrt(T)` (ce n'est plus une vol implicite). `except:` nu. (Module non utilisé, mais à corriger ou supprimer.)

### 3.5 Trois mappings de tickers divergents
`_TICKER_MAP` (`vol_surface.py`), `TICKER_MAP` (`build_vol_surface.py`) et `_ZB_CODES` (`dividends.py`) sont désynchronisés : ex. GLE→`GL4` vs `GL1` ; codes `STM`/`STMPA`, `STLA`/`STLAP` incohérents. Source de bugs silencieux par ticker.

### 3.6 Conventions de day count mélangées
OIS en Act/360 (`_DAY_COUNT = 360`) pour les zéro-taux, maturités produits en Act/365.25, et `_tenor_to_years` approxime un mois à 30 j. Mélange non maîtrisé qui biaise l'actualisation.

### 3.7 I/O réseau caché, fragile et non configurable
Données scrappées en dur depuis `live.euronext.com`, `bluegamma.io`, `zonebourse.com` (parsing HTML très fragile, casse au moindre changement de page) et `yfinance` (`.PA`). Aucune injection de source de données, `verbose=True` par défaut (spam console), gestion des timeouts/retries hétérogène. Le scraping de sites tiers peut en outre poser un problème de conformité (CGU).

### 3.8 État global et effets de bord à l'import
- `analytical.py` modifie `sys.path` **au moment de l'import** (top-level).
- Caches globaux `_SNAPSHOT_CACHE`, `_HESTON_CACHE`, grille `_U_GRID` mutables, **sans éviction ni invalidation temporelle** → en service long, des données de marché périmées peuvent être resservies.

### 3.9 Divers
- `MultiAssetSnapshot` (`market_snapshot.py`, l. 57) contient un `import numpy as np` **dans le corps de la dataclass** — invalide/inutile ; la classe n'est utilisée nulle part.
- `print(...)` partout au lieu du module `logging` ; messages mêlant français et anglais ; emojis dans `build_vol_surface.py`.
- Annotations `"pd.DataFrame"` dans `api.py` sans import de `pandas` au niveau module.
- Imports inutilisés (`math`, `sys`, `os`, `partial`, `timedelta`…).
- `compute_greeks` : convention de signe du theta (`/(-dT_days)`) et gamma via `dS = 1%` à valider (bruité en MC).
- `get_spot` (`build_vol_surface.py`) : `data["Close"].iloc[-1]` peut renvoyer une Series selon la version de yfinance.

---

## Priorisation recommandée

1. **Rendre le projet exécutable** : intégrer (ou réécrire) `Produit_pricing.pdts` dans le dépôt, unifier les imports sous un seul nom de paquet (`pricing_lib/`), ajouter `pyproject.toml` + `requirements.txt`.
2. **Corriger les Greeks** : rho (actualisation OIS bumpée), theta (passer le produit/la maturité), vega (propagation cohérente du bump), Greeks MC barrières.
3. **Ajouter une suite de tests** (parité, convergence, AL vs MC, non-arbitrage) — c'est aussi le meilleur filet pour valider les points §2.
4. **Nettoyer** : supprimer `heston2.py`, `sabr_smoother.py`, doublons de `build_surface` ; unifier les mappings de tickers ; remplacer `print` par `logging` ; ajouter `README` + `LICENSE`.
