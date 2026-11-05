# -*- coding: utf-8 -*-
"""
dreampoly_extended.py

Функции для раздела 11 («Дополнительный статистический анализ») проекта DreamPoly.
Модуль не зависит от переменных ноутбука: все данные передаются аргументами.

Зависимости (добавить в requirements.txt): statsmodels, scipy, seaborn, scikit-learn.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import brentq, linear_sum_assignment
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportion_effectsize,
)

RANDOM_SEED = 42
EMOTIONS_DEFAULT = ['anger', 'fear', 'anticipation', 'trust',
                    'surprise', 'sadness', 'disgust', 'joy']


# =============================================================================
# Общие функции
# =============================================================================

def wilson_ci(k, n):
    """95%-й доверительный интервал Уилсона для доли k/n."""
    if n == 0:
        return np.nan, np.nan
    low, high = proportion_confint(k, n, method='wilson')
    return float(low), float(high)


def prevalence_summary(data, feature):
    """Доля снов с признаком и интервал Уилсона."""
    valid = data[feature].dropna()
    n, k = len(valid), int(valid.sum())
    low, high = wilson_ci(k, n)
    return pd.Series({'n': n, 'count': k,
                      'prevalence': k / n if n else np.nan,
                      'ci_low': low, 'ci_high': high})


def cohens_h(p1, p2):
    """Размер эффекта h Коэна для разности двух долей."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def compare_proportions(k1, n1, k2, n2):
    """
    Сравнение двух долей: разность, ДИ Ньюкомба для разности,
    h Коэна и точный критерий Фишера.
    """
    p1, p2 = k1 / n1, k2 / n2
    odds_ratio, p_value = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    low, high = confint_proportions_2indep(
        k1, n1, k2, n2, method='newcomb', compare='diff')
    return {'p_1': p1, 'p_2': p2, 'diff_1_minus_2': p1 - p2,
            'diff_ci_low': float(low), 'diff_ci_high': float(high),
            'cohens_h': float(cohens_h(p1, p2)),
            'fisher_odds_ratio': float(odds_ratio), 'fisher_p': float(p_value)}


def adjust_pvalues(pvalues, method='fdr_bh'):
    """Поправка на множественные проверки; NaN сохраняются на своих местах."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    if mask.any():
        out[mask] = multipletests(p[mask], method=method)[1]
    return out


def cliffs_delta(x, y):
    """Дельта Клиффа: P(x > y) - P(x < y), через статистику U."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = mannwhitneyu(x, y, alternative='two-sided').statistic
    return 2 * u / (len(x) * len(y)) - 1


def counts_from_rate(rate, token_count):
    """
    Восстанавливает число слов категории по частоте на 1000 токенов.
    В ноутбуке rate = count * 1000 / token_count, поэтому округление точное.
    """
    rate = np.asarray(rate, dtype=float)
    tokens_n = np.asarray(token_count, dtype=float)
    return np.rint(rate * tokens_n / 1000.0).astype(int)


def add_blocks(df, group_col, order_col, block_size=20):
    """
    Нумерует блоки из block_size подряд идущих снов внутри группы.
    Блоки используются как грубый кластер для учёта зависимости
    соседних записей одного дневника.
    """
    d = df.sort_values([group_col, order_col]).copy()
    pos = d.groupby(group_col, observed=True).cumcount()
    d['block'] = d[group_col].astype(str) + '_' + (pos // block_size).astype(str)
    return d


# =============================================================================
# 11.2  Длина текста и лексика гнева: модель частоты с offset
# =============================================================================

def prepare_length_data(izzy, emotion='anger', min_tokens=50, max_tokens=500,
                        block_size=20):
    """
    Готовит данные Izzy: число слов категории, бинарный признак наличия,
    числовой номер сна и блоки для кластеризации.
    """
    d = izzy.copy()
    d = d[(d['token_count'] >= min_tokens) & (d['token_count'] <= max_tokens)].copy()
    count_col = f'{emotion}_count'
    d[count_col] = counts_from_rate(d[emotion], d['token_count'])

    # Проверяем, что частота на 1000 токенов восстанавливается без потерь
    restored = d[count_col] * 1000.0 / d['token_count']
    assert np.abs(restored - d[emotion]).max() < 1e-6, (
        'Частота не соответствует count*1000/token_count')

    d['present'] = (d[count_col] > 0).astype(int)
    d['dream_number'] = pd.to_numeric(d['dream_number_raw'], errors='coerce')
    d = d.dropna(subset=['age_group', 'dream_number'])
    return add_blocks(d, 'age_group', 'dream_number', block_size)


def fit_rate_model(d, count_col, age_ref='18–21', cluster=None):
    """
    Пуассоновская регрессия числа слов категории с offset = log(токенов):
        count ~ возраст + log(token_count),  offset = log(token_count).
    Коэффициент при log(token_count) равен нулю, если частота на слово
    не зависит от длины текста; ненулевой — частота зависит от длины.
    cluster=None -> устойчивые ошибки HC3; иначе кластеры по столбцу cluster.
    """
    formula = (f"{count_col} ~ C(age_group, Treatment(reference='{age_ref}'))"
               " + np.log(token_count)")
    model = smf.glm(formula, data=d, family=sm.families.Poisson(),
                    offset=np.log(d['token_count']))
    if cluster is None:
        return model.fit(cov_type='HC3')
    groups = pd.factorize(d[cluster])[0]
    return model.fit(cov_type='cluster', cov_kwds={'groups': groups})


def rate_model_table(res):
    """Таблица отношений частот (rate ratio) с 95% ДИ."""
    ci = res.conf_int()
    return pd.DataFrame({
        'coef': res.params,
        'rate_ratio': np.exp(res.params),
        'ci_low': np.exp(ci[0]),
        'ci_high': np.exp(ci[1]),
        'p_value': res.pvalues,
    })


def wald_block_test(res, pattern='age_group'):
    """Совместный критерий Вальда для всех коэффициентов, содержащих pattern."""
    names = list(res.params.index)
    target = [n for n in names if pattern in n]
    r_matrix = np.zeros((len(target), len(names)))
    for i, name in enumerate(target):
        r_matrix[i, names.index(name)] = 1.0
    test = res.wald_test(r_matrix, use_f=False)
    return float(np.squeeze(test.statistic)), float(test.pvalue), len(target)


def simulate_null_logit_slope(d, count_col, n_sim=200, seed=RANDOM_SEED):
    """
    Контроль «механического» эффекта длины.
    Если внутри каждого возраста частота слов постоянна, то вероятность
    «есть хотя бы одно слово» растёт с длиной только из-за большего числа слов.
    Функция симулирует такие данные и возвращает наклон логита по log(длина)
    для симуляций и для реальных данных.
    """
    rng = np.random.default_rng(seed)
    sums = d.groupby('age_group', observed=True)[[count_col, 'token_count']].sum()
    rate_by_age = sums[count_col] / sums['token_count']
    lam = d['age_group'].map(rate_by_age).astype(float).to_numpy() * d['token_count'].to_numpy()

    observed = smf.logit('present ~ C(age_group) + np.log(token_count)',
                         data=d).fit(disp=0).params['np.log(token_count)']
    slopes = []
    for _ in range(n_sim):
        sim = d.assign(present=(rng.poisson(lam) > 0).astype(int))
        fit = smf.logit('present ~ C(age_group) + np.log(token_count)',
                        data=sim).fit(disp=0)
        slopes.append(fit.params['np.log(token_count)'])
    return float(observed), np.asarray(slopes)


def length_bin_table(d, count_col, edges=(50, 100, 200, 501)):
    """
    Сравнение возрастов при одинаковой длине: фиксированные интервалы длины,
    общие для всех возрастов. Частота — суммарная (сумма слов / сумма токенов).
    """
    labels = [f'{edges[i]}–{edges[i + 1] - 1}' for i in range(len(edges) - 1)]
    t = d.copy()
    t['length_bin'] = pd.cut(t['token_count'], bins=list(edges), right=False,
                             labels=labels)
    rows = []
    for (age, lbin), g in t.groupby(['age_group', 'length_bin'], observed=True):
        k, n = int(g['present'].sum()), len(g)
        low, high = wilson_ci(k, n)
        rows.append({'age_group': str(age), 'length_bin': str(lbin), 'n_dreams': n,
                     'prevalence': k / n, 'prev_ci_low': low, 'prev_ci_high': high,
                     'rate_per_1000': g[count_col].sum() / g['token_count'].sum() * 1000})
    return pd.DataFrame(rows)


# =============================================================================
# 11.3 / 11.6  Сравнение профилей: тест + размер эффекта + bootstrap по блокам
# =============================================================================

def _block_bootstrap_mean(x, rng, n_boot=2000, block_size=20):
    """Bootstrap среднего блоками подряд идущих наблюдений."""
    x = np.asarray(x, dtype=float)
    blocks = np.array_split(x, max(1, len(x) // block_size))
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks], dtype=float)
    idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    return sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)


def compare_emotion_profiles(df_a, df_b, metrics, label_a='A', label_b='B',
                             n_boot=2000, block_size=20, seed=RANDOM_SEED):
    """
    Для каждой метрики: средние, разность (B - A), критерий Манна–Уитни,
    дельта Клиффа, bootstrap-ДИ разности средних по блокам и поправка BH.
    Строки df_a и df_b должны быть упорядочены по номеру сна.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        x = df_a[metric].dropna().to_numpy(float)
        y = df_b[metric].dropna().to_numpy(float)
        test = mannwhitneyu(x, y, alternative='two-sided')
        boot = (_block_bootstrap_mean(y, rng, n_boot, block_size)
                - _block_bootstrap_mean(x, rng, n_boot, block_size))
        rows.append({
            'metric': metric, f'mean_{label_a}': x.mean(), f'mean_{label_b}': y.mean(),
            'diff_b_minus_a': y.mean() - x.mean(),
            'boot_ci_low': np.quantile(boot, 0.025),
            'boot_ci_high': np.quantile(boot, 0.975),
            'cliffs_delta_a_vs_b': 2 * test.statistic / (len(x) * len(y)) - 1,
            'p_value': test.pvalue, 'n_a': len(x), 'n_b': len(y)})
    out = pd.DataFrame(rows)
    out['p_adjusted_bh'] = adjust_pvalues(out['p_value'])
    return out


def split_halves(df, order_col='dream_number'):
    """Делит упорядоченный по order_col датафрейм на две половины."""
    d = df.sort_values(order_col)
    half = len(d) // 2
    return d.iloc[:half], d.iloc[half:]


def compare_halves_by_group(df, group_col, metrics, order_col='dream_number',
                            min_n=20, seed=RANDOM_SEED, n_boot=1000, block_size=10):
    """
    В каждой группе (например, возрасте) сравнивает вторую половину записей
    с первой по каждой метрике. Поправка BH — по всем сочетаниям группа×метрика.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group, g in df.groupby(group_col, observed=True):
        first, second = split_halves(g, order_col)
        if len(first) < min_n or len(second) < min_n:
            continue
        for metric in metrics:
            x = first[metric].dropna().to_numpy(float)
            y = second[metric].dropna().to_numpy(float)
            if len(x) < min_n or len(y) < min_n:
                continue
            test = mannwhitneyu(x, y, alternative='two-sided')
            boot = (_block_bootstrap_mean(y, rng, n_boot, block_size)
                    - _block_bootstrap_mean(x, rng, n_boot, block_size))
            rows.append({
                'group': str(group), 'metric': metric,
                'n_first': len(x), 'n_second': len(y),
                'mean_first': x.mean(), 'mean_second': y.mean(),
                'diff': y.mean() - x.mean(),
                'boot_ci_low': np.quantile(boot, 0.025),
                'boot_ci_high': np.quantile(boot, 0.975),
                'cliffs_delta': 2 * test.statistic / (len(x) * len(y)) - 1,
                'p_value': test.pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out['p_adjusted_bh'] = adjust_pvalues(out['p_value'])
    return out


def summarize_half_differences(table, alpha=0.05):
    """Итог по метрике: доля групп с ростом и число значимых различий после BH."""
    rows = []
    for metric, g in table.groupby('metric', sort=False):
        rows.append({'metric': metric, 'n_groups': len(g),
                     'share_increased': (g['diff'] > 0).mean(),
                     'n_significant_bh': int((g['p_adjusted_bh'] < alpha).sum()),
                     'median_abs_cliffs_delta': g['cliffs_delta'].abs().median()})
    return pd.DataFrame(rows)


# =============================================================================
# 11.5  Совместная встречаемость категорий NRC
# =============================================================================

def add_emotion_presence(df, emotions=EMOTIONS_DEFAULT):
    """Добавляет столбцы <emotion>_count и <emotion>_present."""
    d = df.copy()
    for emotion in emotions:
        d[f'{emotion}_count'] = counts_from_rate(d[emotion], d['token_count'])
        d[f'{emotion}_present'] = (d[f'{emotion}_count'] > 0).astype(int)
    return d


def length_strata(token_count, edges=(0, 50, 100, 200, np.inf)):
    """Страты длины текста для поправки на длину."""
    labels = ['<50', '50–99', '100–199', '≥200']
    return pd.cut(token_count, bins=list(edges), right=False, labels=labels)


def cooccurrence_table(df, emotions=EMOTIONS_DEFAULT, presence_suffix='_present',
                       strata_col='length_stratum'):
    """
    Для каждой пары категорий: распространённости, lift, максимально возможный
    lift, сырое отношение шансов и критерий Фишера, а также отношение шансов
    Мантеля–Хэнзеля с поправкой на страты длины.
    """
    rows = []
    for a, b in combinations(emotions, 2):
        xa = df[a + presence_suffix].to_numpy(bool)
        xb = df[b + presence_suffix].to_numpy(bool)
        n = len(df)
        p_a, p_b, p_both = xa.mean(), xb.mean(), (xa & xb).mean()
        table = [[int((xa & xb).sum()), int((xa & ~xb).sum())],
                 [int((~xa & xb).sum()), int((~xa & ~xb).sum())]]
        raw_or, raw_p = fisher_exact(table)

        tables = []
        for _, g in df.groupby(strata_col, observed=True):
            ga, gb = g[a + presence_suffix].to_numpy(bool), g[b + presence_suffix].to_numpy(bool)
            tables.append(np.array([[(ga & gb).sum(), (ga & ~gb).sum()],
                                    [(~ga & gb).sum(), (~ga & ~gb).sum()]]))
        try:
            strat = StratifiedTable(tables)
            mh_or = float(strat.oddsratio_pooled)
            mh_low, mh_high = strat.oddsratio_pooled_confint()
            mh_p = float(strat.test_null_odds(correction=False).pvalue)
        except Exception:
            mh_or = mh_low = mh_high = mh_p = np.nan

        expected = p_a * p_b
        rows.append({
            'emotion_a': a, 'emotion_b': b, 'n_dreams': n,
            'prevalence_a': p_a, 'prevalence_b': p_b, 'joint_prevalence': p_both,
            'lift': p_both / expected if expected > 0 else np.nan,
            'max_possible_lift': min(p_a, p_b) / expected if expected > 0 else np.nan,
            'raw_odds_ratio': float(raw_or), 'raw_fisher_p': float(raw_p),
            'mh_odds_ratio': mh_or, 'mh_ci_low': float(mh_low), 'mh_ci_high': float(mh_high),
            'mh_p': mh_p})
    out = pd.DataFrame(rows)
    out['raw_p_adjusted_bh'] = adjust_pvalues(out['raw_fisher_p'])
    out['mh_p_adjusted_bh'] = adjust_pvalues(out['mh_p'])
    return out


def odds_ratio_matrix(table, value_col, emotions=EMOTIONS_DEFAULT):
    """Симметричная матрица значений value_col по парам категорий."""
    m = pd.DataFrame(np.nan, index=emotions, columns=emotions, dtype=float)
    for _, row in table.iterrows():
        m.loc[row['emotion_a'], row['emotion_b']] = row[value_col]
        m.loc[row['emotion_b'], row['emotion_a']] = row[value_col]
    return m


def get_nrc_lexicon(nrc_obj):
    """Достаёт словарь слово -> список эмоций из объекта NRCLex."""
    for attr in ('__lexicon__', 'lexicon'):
        lexicon = getattr(nrc_obj, attr, None)
        if isinstance(lexicon, dict) and lexicon:
            return lexicon
    raise AttributeError(
        'Не удалось получить словарь из объекта NRCLex; '
        'проверьте атрибуты dir(nrc) в вашей версии пакета nrclex.')


def exclusive_emotion_counts(words, lexicon, emotions=EMOTIONS_DEFAULT):
    """
    Считает только слова, которые в NRC относятся ровно к одной из восьми
    категорий. Убирает «искусственную» совместную встречаемость, когда одно
    слово записано сразу в несколько категорий.
    """
    allowed = set(emotions)
    counts = dict.fromkeys(emotions, 0)
    for word in words:
        cats = set(lexicon.get(word, [])) & allowed
        if len(cats) == 1:
            counts[next(iter(cats))] += 1
    return counts


# =============================================================================
# 11.1 / 11.4  Лексические маркеры агрессии и сравнение серий
# =============================================================================

def _forms(stem_forms):
    return r'\b(?:' + '|'.join(stem_forms) + r')\b'


# Широкий словарь: исходный список из первой версии раздела
AGGRESSION_PATTERNS_BROAD = {
    'attack': _forms(['attack', 'attacks', 'attacked', 'attacking']),
    'kill': _forms(['kill', 'kills', 'killed', 'killing']),
    'murder': _forms(['murder', 'murders', 'murdered', 'murdering']),
    'fight': _forms(['fight', 'fights', 'fighting', 'fought']),
    'beat': _forms(['beat', 'beats', 'beating', 'beaten']),
    'hit': _forms(['hit', 'hits', 'hitting']),
    'hurt': _forms(['hurt', 'hurts', 'hurting']),
    'stab': _forms(['stab', 'stabs', 'stabbed', 'stabbing']),
    'shoot': _forms(['shoot', 'shoots', 'shooting', 'shot']),
    'strangle': _forms(['strangle', 'strangles', 'strangled', 'strangling']),
    'choke': _forms(['choke', 'chokes', 'choked', 'choking']),
    'punch': _forms(['punch', 'punches', 'punched', 'punching']),
    'kick': _forms(['kick', 'kicks', 'kicked', 'kicking']),
    'slap': _forms(['slap', 'slaps', 'slapped', 'slapping']),
    'bite': _forms(['bite', 'bites', 'bit', 'bitten', 'biting']),
    'wound': _forms(['wound', 'wounds', 'wounded', 'wounding']),
    'injure': _forms(['injure', 'injures', 'injured', 'injuring']),
    'harm': _forms(['harm', 'harms', 'harmed', 'harming']),
    'threaten': _forms(['threaten', 'threatens', 'threatened', 'threatening']),
    'assault': _forms(['assault', 'assaults', 'assaulted', 'assaulting']),
    'smash': _forms(['smash', 'smashes', 'smashed', 'smashing']),
    'crush': _forms(['crush', 'crushes', 'crushed', 'crushing']),
}

# Строгий словарь: убираем самые частые неагрессивные значения
#  - «a bit» не считаем формой bite;
#  - «shot» считаем только после слов, указывающих на выстрел в кого-то;
#  - «a crush» (влюблённость) не считаем.
_SHOT_CONTEXT = (r'(?:was|were|got|get|gets|been|be|is|am|he|she|they|i|we|'
                 r'someone|him|her|me|them|us)')
AGGRESSION_PATTERNS_STRICT = dict(AGGRESSION_PATTERNS_BROAD)
AGGRESSION_PATTERNS_STRICT['bite'] = _forms(['bite', 'bites', 'bitten', 'biting'])
AGGRESSION_PATTERNS_STRICT['shoot'] = (
    r'\b(?:shoot|shoots|shooting)\b|\b' + _SHOT_CONTEXT + r'\s+shot\b')
AGGRESSION_PATTERNS_STRICT['crush'] = (
    r'(?<!\ba )(?<!\bhave a )(?<!\bhad a )\b(?:crush|crushes|crushed|crushing)\b')


def compile_patterns(patterns):
    return {k: re.compile(v, flags=re.IGNORECASE) for k, v in patterns.items()}


def add_marker_columns(data, patterns, prefix, text_col='text',
                       token_col='token_count'):
    """
    Добавляет <prefix>_count, <prefix>_present, <prefix>_per1000.
    Возвращает (датафрейм, таблица счётчиков по отдельным маркерам).
    """
    compiled = compile_patterns(patterns)
    lowered = data[text_col].fillna('').astype(str).str.lower()
    per_marker = pd.DataFrame(
        {name: lowered.apply(lambda t, p=pat: len(p.findall(t)))
         for name, pat in compiled.items()}, index=data.index)
    out = data.copy()
    out[f'{prefix}_count'] = per_marker.sum(axis=1)
    out[f'{prefix}_present'] = (out[f'{prefix}_count'] > 0).astype(int)
    out[f'{prefix}_per1000'] = np.where(
        out[token_col] > 0, out[f'{prefix}_count'] / out[token_col] * 1000, np.nan)
    return out, per_marker


def kwic_sample(data, patterns, n_per_marker=20, window=50, seed=RANDOM_SEED,
                text_col='text'):
    """
    Выборка контекстов (KWIC) для проверки точности словаря: по n_per_marker
    случайных совпадений на каждый маркер. Это небольшой аудит словаря,
    а не ручная разметка корпуса.
    """
    rng = np.random.default_rng(seed)
    compiled = compile_patterns(patterns)
    rows = []
    for name, pat in compiled.items():
        hits = []
        for dream_id, series_id, text in zip(
                data['dream_id'], data['series_id'], data[text_col].fillna('')):
            for m in pat.finditer(str(text)):
                start, end = max(0, m.start() - window), min(len(text), m.end() + window)
                hits.append((dream_id, series_id, text[start:end].replace('\n', ' '),
                             m.group(0)))
        if not hits:
            continue
        pick = rng.choice(len(hits), size=min(n_per_marker, len(hits)), replace=False)
        for i in pick:
            dream_id, series_id, context, matched = hits[i]
            rows.append({'marker': name, 'matched': matched, 'dream_id': dream_id,
                         'series_id': series_id, 'context': context,
                         'is_aggression_context': ''})
    return pd.DataFrame(rows)


def summarize_by_series(data, flag_col, count_col, token_col='token_count'):
    """Сводка по сериям: доля снов с признаком, ДИ Уилсона, частота на 1000 токенов."""
    rows = []
    for series_id, g in data.groupby('series_id'):
        n, k = len(g), int(g[flag_col].sum())
        low, high = wilson_ci(k, n)
        tokens_sum = int(g[token_col].sum())
        markers_sum = int(g[count_col].sum())
        rows.append({'series_id': series_id, 'n_dreams': n, 'dreams_with_marker': k,
                     'prevalence': k / n, 'ci_low': low, 'ci_high': high,
                     'mean_tokens': g[token_col].mean(),
                     'markers_per_1000': markers_sum / tokens_sum * 1000 if tokens_sum else np.nan})
    return pd.DataFrame(rows).sort_values('prevalence', ascending=False).reset_index(drop=True)


def compare_two_series(data, series_a, series_b, flag_col, count_col,
                       token_col='token_count'):
    """
    Сравнение series_a и series_b (разность = a - b):
    1) сырая разность долей, ДИ Ньюкомба, h Коэна, Фишер;
    2) логит «есть маркер» с поправкой на log(длины);
    3) пуассоновская модель числа маркеров с offset = log(токенов).
    Ряды считаются независимыми; парность не учитывается.
    """
    ga = data[data['series_id'] == series_a]
    gb = data[data['series_id'] == series_b]
    res = {'series_a': series_a, 'series_b': series_b,
           'mean_tokens_a': ga[token_col].mean(), 'mean_tokens_b': gb[token_col].mean()}
    res.update(compare_proportions(int(ga[flag_col].sum()), len(ga),
                                   int(gb[flag_col].sum()), len(gb)))
    res['n_a'], res['n_b'] = len(ga), len(gb)

    d = data[data['series_id'].isin([series_a, series_b]) & (data[token_col] > 0)].copy()
    d['is_a'] = (d['series_id'] == series_a).astype(int)
    d['present_'] = d[flag_col].astype(int)
    d['count_'] = d[count_col].astype(int)
    d['tok_'] = d[token_col]
    try:
        logit = smf.logit('present_ ~ is_a + np.log(tok_)', data=d).fit(disp=0, cov_type='HC3')
        low, high = np.exp(logit.conf_int().loc['is_a'])
        res.update({'adj_or_a_vs_b': float(np.exp(logit.params['is_a'])),
                    'adj_or_ci_low': float(low), 'adj_or_ci_high': float(high),
                    'adj_or_p': float(logit.pvalues['is_a'])})
    except Exception:
        res.update({'adj_or_a_vs_b': np.nan, 'adj_or_ci_low': np.nan,
                    'adj_or_ci_high': np.nan, 'adj_or_p': np.nan})
    try:
        pois = smf.glm('count_ ~ is_a', data=d, family=sm.families.Poisson(),
                       offset=np.log(d['tok_'])).fit(cov_type='HC3')
        low, high = np.exp(pois.conf_int().loc['is_a'])
        res.update({'rate_ratio_a_vs_b': float(np.exp(pois.params['is_a'])),
                    'rr_ci_low': float(low), 'rr_ci_high': float(high),
                    'rr_p': float(pois.pvalues['is_a'])})
    except Exception:
        res.update({'rate_ratio_a_vs_b': np.nan, 'rr_ci_low': np.nan,
                    'rr_ci_high': np.nan, 'rr_p': np.nan})
    return res


# =============================================================================
# 11.8  Статистическая мощность и минимально обнаружимая разность
# =============================================================================

def mde_proportion(n1, n2, base_rate, power=0.8, alpha=0.05, deff=1.0):
    """
    Минимальная разность долей (group2 - group1), обнаруживаемая с заданной
    мощностью. deff — эффект дизайна: эффективные n = n / deff.
    Возвращает NaN, если при таком объёме мощность недостижима.
    """
    n1e, n2e = n1 / deff, n2 / deff
    analysis = NormalIndPower()

    def gap(delta):
        h = proportion_effectsize(base_rate + delta, base_rate)
        return analysis.power(effect_size=h, nobs1=n1e, ratio=n2e / n1e,
                              alpha=alpha, alternative='two-sided') - power

    upper = 1 - base_rate - 1e-4
    if upper <= 1e-4 or gap(upper) < 0:
        return np.nan
    return float(brentq(gap, 1e-4, upper))


def mde_table(designs, base_rates, power=0.8, alpha=0.05, deffs=(1.0,)):
    """Таблица MDE для набора дизайнов {название: (n1, n2)}."""
    rows = []
    for name, (n1, n2) in designs.items():
        for deff in deffs:
            for base in base_rates:
                rows.append({'design': name, 'n1': n1, 'n2': n2, 'deff': deff,
                             'base_rate': base,
                             'mde': mde_proportion(n1, n2, base, power, alpha, deff)})
    return pd.DataFrame(rows)


def simulate_fisher_power(n1, n2, p1, p2, n_sim=2000, alpha=0.05, seed=RANDOM_SEED):
    """Мощность точного критерия Фишера методом Монте-Карло."""
    rng = np.random.default_rng(seed)
    k1 = rng.binomial(n1, p1, size=n_sim)
    k2 = rng.binomial(n2, p2, size=n_sim)
    hits = sum(fisher_exact([[a, n1 - a], [b, n2 - b]])[1] < alpha
               for a, b in zip(k1, k2))
    return hits / n_sim


# =============================================================================
# 11.7  Имена собственные и тематическая модель
# =============================================================================

DEFAULT_NAME_EXCLUDE = {
    # родство и обращения: это содержательные слова, а не имена
    'mom', 'dad', 'mommy', 'daddy', 'nana', 'poppa', 'grandma', 'grandpa',
    'grandmother', 'grandfather', 'mother', 'father', 'aunt', 'uncle', 'god',
    'mr', 'mrs', 'ms', 'dr', 'miss', 'sir',
    # дни недели, месяцы, праздники
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    'christmas', 'halloween', 'thanksgiving', 'easter',
}

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def find_person_names(texts, min_count=5, min_ratio=0.9, exclude=DEFAULT_NAME_EXCLUDE):
    """
    Ищет слова, которые почти всегда пишутся с заглавной буквы НЕ в начале
    предложения. Результат нужно просмотреть вручную: в него попадут и
    географические названия.
    """
    capital_mid, lower_any = Counter(), Counter()
    for text in texts:
        text = str(text)
        for m in _TOKEN_RE.finditer(text):
            token = m.group(0)
            # Убираем притяжательное 's: «Ezra's» считаем как «ezra»
            word = re.sub(r"['’]s$", '', token.lower())
            if token[0].islower():
                lower_any[word] += 1
                continue
            if token == 'I' or token.startswith(("I'", 'I’')):
                continue
            if token.isupper() and len(token) > 1:  # аббревиатуры: TV, USA
                continue
            prefix = text[:m.start()].rstrip(' "\'“”‘’()')
            if not prefix or prefix[-1] in '.!?':   # начало предложения
                continue
            capital_mid[word] += 1
    rows = []
    for word, cap in capital_mid.items():
        low = lower_any.get(word, 0)
        ratio = cap / (cap + low)
        if cap >= min_count and ratio >= min_ratio and len(word) > 2 and word not in exclude:
            rows.append({'word': word, 'capital_mid_count': cap,
                         'lower_count': low, 'capital_ratio': ratio})
    return (pd.DataFrame(rows, columns=['word', 'capital_mid_count', 'lower_count',
                                        'capital_ratio'])
            .sort_values('capital_mid_count', ascending=False).reset_index(drop=True))


def fit_nmf_topics(texts, extra_stop_words=(), n_topics=8, seed=RANDOM_SEED):
    """
    TF-IDF + NMF с теми же параметрами, что и в разделе 8 ноутбука.
    С extra_stop_words=() результат совпадает с исходной моделью.
    """
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | set(extra_stop_words))
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2),
                                 min_df=10, max_df=0.85, max_features=6000,
                                 sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    model = NMF(n_components=n_topics, init='nndsvda', random_state=seed, max_iter=700)
    weights = model.fit_transform(matrix)
    return {'terms': vectorizer.get_feature_names_out(), 'H': model.components_,
            'W': weights, 'model': model}


def topic_top_terms(fit, n_terms=12):
    return [', '.join(fit['terms'][row.argsort()[-n_terms:][::-1]]) for row in fit['H']]


def topic_shares_by_group(fit, groups):
    """Средний нормированный вес каждой темы по группам (как в разделе 8)."""
    shares = fit['W'] / np.maximum(fit['W'].sum(axis=1, keepdims=True), 1e-12)
    table = pd.DataFrame(shares, columns=[f'topic_{i + 1}' for i in range(shares.shape[1])])
    table['group'] = np.asarray(groups)
    return table.groupby('group', observed=True).mean()


def topic_index_by_term(fit, term):
    """Номер темы (с нуля), где слово term имеет наибольший вес."""
    terms = list(fit['terms'])
    if term not in terms:
        raise KeyError(f'Слово «{term}» не вошло в словарь модели')
    return int(np.argmax(fit['H'][:, terms.index(term)]))


def match_topics(fit_a, fit_b):
    """Сопоставляет темы двух моделей по косинусной близости (венгерский алгоритм)."""
    terms_a, terms_b = list(fit_a['terms']), list(fit_b['terms'])
    common = sorted(set(terms_a) & set(terms_b))
    ia = [terms_a.index(t) for t in common]
    ib = [terms_b.index(t) for t in common]
    a, b = fit_a['H'][:, ia], fit_b['H'][:, ib]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    rows, cols = linear_sum_assignment(-sim)
    return pd.DataFrame({'topic_a': rows + 1, 'topic_b': cols + 1,
                         'cosine': sim[rows, cols]})
