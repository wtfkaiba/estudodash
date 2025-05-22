import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_extras.metric_cards import style_metric_cards # Você ainda usa isso

# Configuração inicial da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(layout="wide", page_title="Dashboard de Investimentos")

# --- Constantes e Configurações Globais ---
IMAGES_BASE_URL = "https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/"
LOGO_ACOES_PATH = "images/logo.png" # Certifique-se que este caminho é válido
LOGO_FIIS_PATH = "images/fiilogo.png"   # Certifique-se que este caminho é válido

# --- Funções Auxiliares ---

@st.cache_data(ttl=300) # Cache por 5 minutos para dados de preço
def load_price_data(_tickers_with_suffix, _benchmark_ticker, _start_date, _end_date, data_label="Ativos"):
    """
    Baixa dados de preços para uma lista de tickers e um benchmark.
    Retorna um DataFrame com os preços de fechamento (coluna 'Close' do yfinance, que é ajustada por padrão)
    e o benchmark. As colunas são renomeadas (sem '.SA', '^BVSP' para 'IBOV').
    Retorna DataFrame vazio em caso de falha ou nenhum dado.
    """
    all_tickers_to_download = []
    if _tickers_with_suffix:
        all_tickers_to_download.extend(list(set(_tickers_with_suffix)))

    if _benchmark_ticker and _benchmark_ticker not in all_tickers_to_download:
        all_tickers_to_download.append(_benchmark_ticker)

    if not all_tickers_to_download:
        return pd.DataFrame()

    try:
        raw_data = yf.download(all_tickers_to_download, start=_start_date, end=_end_date, progress=False, ignore_tz=True)

        if raw_data.empty:
            return pd.DataFrame()

        if isinstance(raw_data.columns, pd.MultiIndex):
            prices_df = raw_data['Close'].copy()
        else:
            if 'Close' in raw_data.columns:
                prices_df = raw_data[['Close']].copy()
                if len(all_tickers_to_download) == 1:
                    prices_df.columns = [all_tickers_to_download[0]]
            else:
                return pd.DataFrame()
        
        if isinstance(prices_df, pd.Series):
            prices_df = prices_df.to_frame(name=prices_df.name or all_tickers_to_download[0])

        final_columns = {}
        for col_name in prices_df.columns:
            new_name = str(col_name).replace(".SA", "")
            if new_name == "^BVSP": new_name = "IBOV"
            final_columns[col_name] = new_name
        prices_df = prices_df.rename(columns=final_columns)
        
        # Adicionado para depuração - Quais colunas estão realmente aqui?
        # st.caption(f"Debug load_price_data for {data_label}: Colunas após renomear: {list(prices_df.columns)}")

        if _tickers_with_suffix:
            expected_main_tickers_clean = {str(t).replace(".SA", "") for t in _tickers_with_suffix}
            missing_main = expected_main_tickers_clean - set(prices_df.columns)
            if missing_main:
                st.caption(f"Aviso em {data_label}: Dados não encontrados para: {', '.join(sorted(list(missing_main)))}.")
        
        if _benchmark_ticker:
            clean_benchmark_name = _benchmark_ticker.replace(".SA", "")
            if clean_benchmark_name == "^BVSP": clean_benchmark_name = "IBOV"
            if clean_benchmark_name not in prices_df.columns:
                prices_df[clean_benchmark_name] = np.nan
        
        return prices_df.dropna(how='all', axis=0)

    except Exception as e:
        st.error(f"Erro ao baixar/processar dados para {data_label}: {str(e)[:200]}")
        return pd.DataFrame()


@st.cache_data(ttl=900) # Cache por 15 minutos para dados de info e cotação
def get_ticker_fundamental_info(ticker_symbol_with_suffix):
    """
    Busca informações 'info' e a cotação mais recente para um ticker.
    Retorna um dicionário com dados selecionados ou None se houver erro.
    """
    try:
        ticker_obj = yf.Ticker(ticker_symbol_with_suffix)
        info = ticker_obj.info

        selected_info = {
            "Nome Curto": info.get("shortName"),
            "Símbolo": info.get("symbol"),
            "Setor": info.get("sector"),
            "Indústria": info.get("industry"),
            "País": info.get("country"),
            "Website": info.get("website"),
            "Resumo do Negócio": info.get("longBusinessSummary"),
            "Preço Atual": info.get("currentPrice") or info.get("regularMarketPrice"),
            "Máxima do Dia": info.get("dayHigh"),
            "Mínima do Dia": info.get("dayLow"),
            "Variação (%) Dia": info.get("regularMarketChangePercent", 0) * 100 if info.get("regularMarketChangePercent") else None,
            "Volume": info.get("volume"),
            "Valor de Mercado": info.get("marketCap"),
            "P/L (TTM)": info.get("trailingPE"),
            "P/VP": info.get("priceToBook"),
            "Dividend Yield (%)": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
            "ROE (TTM) (%)": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
            "Beta": info.get("beta"),
            "Próxima Data Ex-Dividendo": pd.to_datetime(info.get("exDividendDate"), unit='s').strftime('%d/%m/%Y') if info.get("exDividendDate") else None,
            "Recomendação Média": info.get("recommendationKey"),
            "Preço Alvo Médio": info.get("targetMeanPrice"),
        }
        
        for key, value in selected_info.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                selected_info[key] = None
            elif value == "N/A":
                 selected_info[key] = None

        return {k: v for k, v in selected_info.items() if v is not None}

    except Exception: # Captura exceções mais genéricas do yfinance para .info
        # st.caption(f"Info não disponível para {ticker_symbol_with_suffix}.")
        return None


def display_fundamental_details(ticker_name_clean, ticker_info_dict):
    """Exibe os detalhes fundamentalistas e de mercado para um ticker."""
    
    st.subheader(f"Detalhes de: {ticker_info_dict.get('Nome Curto', ticker_name_clean)} ({ticker_name_clean})")
    cols_count = 2
    
    price_col, market_col = st.columns(cols_count)
    with price_col:
        current_price = ticker_info_dict.get('Preço Atual')
        st.metric(label="Preço Atual", value=f"R$ {current_price:,.2f}" if current_price is not None else "N/A")
    with market_col:
        var_dia = ticker_info_dict.get('Variação (%) Dia')
        st.metric(label="Variação Dia (%)", value=f"{var_dia:.2f}%" if var_dia is not None else "N/A", 
                  delta=f"{var_dia:.2f}%" if var_dia is not None and var_dia != 0 else None)

    st.markdown("##### Mercado")
    market_info_cols = st.columns(cols_count)
    market_info_data = {
        "Máxima Dia": ticker_info_dict.get('Máxima do Dia'),
        "Mínima Dia": ticker_info_dict.get('Mínima do Dia'),
        "Volume (un.)": ticker_info_dict.get('Volume'),
        "Valor de Mercado": ticker_info_dict.get('Valor de Mercado'),
    }
    current_col_idx = 0
    for label, value in market_info_data.items():
        if value is not None:
            with market_info_cols[current_col_idx % cols_count]:
                if isinstance(value, (float, int)) and value > 10000:
                     st.markdown(f"**{label}:** {value:,.0f}")
                elif isinstance(value, float):
                     st.markdown(f"**{label}:** {value:,.2f}")
                else:
                     st.markdown(f"**{label}:** {value}")
            current_col_idx += 1
    if current_col_idx == 0: st.caption("Nenhuma informação de mercado adicional disponível.")
    st.markdown("---")

    st.markdown("##### Indicadores Fundamentalistas")
    fundamental_cols = st.columns(cols_count)
    fundamental_data = {
        "P/L (TTM)": ticker_info_dict.get('P/L (TTM)'),
        "P/VP": ticker_info_dict.get('P/VP'),
        "Dividend Yield Anualizado (%)": ticker_info_dict.get('Dividend Yield (%)'),
        "ROE (TTM) (%)": ticker_info_dict.get('ROE (TTM) (%)'),
        "Beta": ticker_info_dict.get('Beta'),
        "Próxima Data Ex-Dividendo": ticker_info_dict.get('Próxima Data Ex-Dividendo'),
    }
    current_col_idx = 0
    for label, value in fundamental_data.items():
        if value is not None:
            with fundamental_cols[current_col_idx % cols_count]:
                if " (%)" in label and isinstance(value, (float, int)):
                    st.markdown(f"**{label.replace(' (%)','').strip()}:** {value:.2f}%")
                elif isinstance(value, float):
                     st.markdown(f"**{label}:** {value:.2f}")
                else:
                    st.markdown(f"**{label}:** {value}")
            current_col_idx +=1
    if current_col_idx == 0: st.caption("Nenhum indicador fundamentalista principal disponível.")
    st.markdown("---")
    
    st.markdown("##### Sobre a Empresa")
    company_info_cols = st.columns(cols_count)
    company_data = {
        "Setor": ticker_info_dict.get('Setor'),
        "Indústria": ticker_info_dict.get('Indústria'),
        "País": ticker_info_dict.get('País'),
        "Website": ticker_info_dict.get('Website'),
        "Recomendação Média": ticker_info_dict.get('Recomendação Média'),
        "Preço Alvo Médio": ticker_info_dict.get('Preço Alvo Médio'),
    }
    current_col_idx = 0
    for label, value in company_data.items():
        if value is not None:
            with company_info_cols[current_col_idx % cols_count]:
                if label == "Website" and value and ("http" in value or "www" in value):
                    st.markdown(f"**{label}:** [{value}]({value if value.startswith('http') else 'http://'+value})")
                elif isinstance(value, (float, int)) and value > 1000:
                     st.markdown(f"**{label}:** {value:,.2f}")
                else:
                    st.markdown(f"**{label}:** {value}")
            current_col_idx +=1
    if current_col_idx == 0: st.caption("Nenhuma informação adicional sobre a empresa disponível.")
            
    business_summary = ticker_info_dict.get("Resumo do Negócio")
    if business_summary:
        with st.expander("Resumo do Negócio", expanded=False):
            st.write(business_summary)
    else:
        st.caption("Resumo do negócio não disponível.")


def build_sidebar_generic(section_title, csv_file_path, multiselect_label, multiselect_placeholder, date_input_key_prefix, image_path=None):
    if image_path:
        try: st.image(image_path, width=60)
        except Exception: st.caption(f"Logo {section_title} indisponível.")
            
    st.subheader(section_title)
    
    try:
        ticker_list_df = pd.read_csv(csv_file_path, header=None)
        if not ticker_list_df.empty and ticker_list_df.shape[1] > 1:
            options = sorted(list(set(ticker_list_df.iloc[:, 1].astype(str).str.strip().tolist())))
        else:
            st.warning(f"Arquivo {csv_file_path} vazio ou formato incorreto.")
            options = []
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {csv_file_path}")
        options = []
    except Exception as e:
        st.error(f"Erro ao carregar {csv_file_path}: {e}")
        options = []

    selected_tickers_no_suffix = st.multiselect(
        label=multiselect_label, options=options, placeholder=multiselect_placeholder,
        key=f"{date_input_key_prefix}_multiselect"
    )
    selected_tickers_with_suffix = [f"{t}.SA" for t in selected_tickers_no_suffix]
    
    today = datetime.now()
    default_start_date = datetime(today.year, 1, 1) if today.month > 1 or today.day > 2 else datetime(today.year -1, 1,1)
    
    start_date = st.date_input("De", format="DD/MM/YYYY", value=default_start_date, key=f"{date_input_key_prefix}_start_date")
    end_date = st.date_input("Até", format="DD/MM/YYYY", value=today, key=f"{date_input_key_prefix}_end_date")

    if start_date > end_date:
        st.error("Data 'De' posterior à data 'Até'. Ajustando para padrões.")
        start_date, end_date = default_start_date, today

    return selected_tickers_no_suffix, selected_tickers_with_suffix, start_date, end_date


def build_main_section(
    section_title_display, selected_tickers_clean, prices_df_full,
    benchmark_col_clean, portfolio_col_label, images_config
):
    if prices_df_full is None or prices_df_full.empty:
        return

    main_tickers_present = [tc for tc in selected_tickers_clean if tc in prices_df_full.columns]

    cols_for_analysis = main_tickers_present[:]
    if benchmark_col_clean in prices_df_full.columns:
        cols_for_analysis.append(benchmark_col_clean)
    else: # Adicionado para depuração
        st.caption(f"Benchmark '{benchmark_col_clean}' não encontrado em prices_df_full para {section_title_display}. Colunas disponíveis: {list(prices_df_full.columns)}")


    analysis_df = prices_df_full[list(set(cols_for_analysis))].copy()
    analysis_df = analysis_df.dropna(axis=0, how='any')

    if analysis_df.empty or len(analysis_df) < 2:
        st.warning(f"Dados insuficientes para {section_title_display} após limpeza ou período curto.")
        return
        
    portfolio_tickers_in_analysis = [mt for mt in main_tickers_present if mt in analysis_df.columns]
    if portfolio_tickers_in_analysis:
        weights = np.ones(len(portfolio_tickers_in_analysis)) / len(portfolio_tickers_in_analysis)
        analysis_df[portfolio_col_label] = analysis_df[portfolio_tickers_in_analysis].dot(weights)
    else:
        analysis_df[portfolio_col_label] = np.nan

    norm_prices = 100 * analysis_df / analysis_df.iloc[0]
    returns = analysis_df.pct_change().iloc[1:]
    
    if returns.empty:
        vols = pd.Series(0.0, index=analysis_df.columns); rets_total_period = pd.Series(0.0, index=analysis_df.columns); sharpe_ratios = pd.Series(0.0, index=analysis_df.columns)
    else:
        vols = returns.std() * np.sqrt(252)
        rets_total_period = (norm_prices.iloc[-1] / norm_prices.iloc[0]) - 1
        mean_daily_returns = returns.mean(); std_daily_returns = returns.std()
        sharpe_ratios = (mean_daily_returns / std_daily_returns) * np.sqrt(252); sharpe_ratios = sharpe_ratios.fillna(0)

    st.subheader(f"Dashboard: {section_title_display}")

    ordered_cols_for_cards = []
    if portfolio_col_label in analysis_df.columns and not analysis_df[portfolio_col_label].isnull().all():
        ordered_cols_for_cards.append(portfolio_col_label)
    if benchmark_col_clean in analysis_df.columns and not analysis_df[benchmark_col_clean].isnull().all() :
        ordered_cols_for_cards.append(benchmark_col_clean)
    for col in portfolio_tickers_in_analysis:
        if col not in ordered_cols_for_cards: ordered_cols_for_cards.append(col)
    
    num_metrics_display_cols = min(len(ordered_cols_for_cards), 5)
    if num_metrics_display_cols > 0:
        metric_grid_cols = st.columns(num_metrics_display_cols)
        col_idx = 0
        for t_col_name in ordered_cols_for_cards:
            if t_col_name not in analysis_df.columns: continue
            current_metric_col = metric_grid_cols[col_idx % num_metrics_display_cols]
            with current_metric_col:
                with st.container(border=True):
                    st.subheader(t_col_name, divider="red")
                    icon_path_or_url = None
                    if t_col_name == portfolio_col_label: icon_path_or_url = images_config.get('portfolio')
                    elif t_col_name == benchmark_col_clean: icon_path_or_url = images_config.get('benchmark')
                    else:
                        base_url = images_config.get('default_ticker_icon_base_url')
                        if base_url: icon_path_or_url = f'{base_url}{t_col_name}.png'
                    if icon_path_or_url:
                        try: st.image(icon_path_or_url, width=60)
                        except Exception: st.caption(f"Ícone {t_col_name} indisponível")
                    
                    ret_val = rets_total_period.get(t_col_name, 0.0); vol_val = vols.get(t_col_name, 0.0); sharpe_val = sharpe_ratios.get(t_col_name, 0.0)
                    st.metric(label="Retorno Total", value=f"{ret_val:.2%}")
                    st.metric(label="Volatilidade Anual.", value=f"{vol_val:.2%}")
                    st.metric(label="Índice Sharpe Anual.", value=f"{sharpe_val:.2f}")
            col_idx +=1
        style_metric_cards(background_color='rgba(255,255,255,0)', border_left_color='rgba(200,50,50,0.8)')
    else:
        st.info(f"Sem dados suficientes para exibir métricas de {section_title_display.lower()}.")

    # Detalhes Fundamentalistas
    individual_tickers_for_details = [tc for tc in portfolio_tickers_in_analysis if tc != portfolio_col_label and tc != benchmark_col_clean]
    if individual_tickers_for_details:
        st.markdown("---"); st.subheader(f"Análise Detalhada dos {section_title_display}")
        if len(individual_tickers_for_details) == 1:
            ticker_name_clean = individual_tickers_for_details[0]
            ticker_info = get_ticker_fundamental_info(f"{ticker_name_clean}.SA")
            if ticker_info: display_fundamental_details(ticker_name_clean, ticker_info)
            else: st.warning(f"Detalhes não carregados para {ticker_name_clean}.")
        else:
            tab_labels = individual_tickers_for_details
            tabs = st.tabs(tab_labels)
            for i, tab_widget in enumerate(tabs):
                with tab_widget:
                    ticker_name_clean = tab_labels[i]
                    ticker_info = get_ticker_fundamental_info(f"{ticker_name_clean}.SA")
                    if ticker_info: display_fundamental_details(ticker_name_clean, ticker_info)
                    else: st.warning(f"Detalhes não carregados para {ticker_name_clean}.")
    
    st.markdown("---") # Divisor antes dos gráficos
    chart_col1, chart_col2 = st.columns(2, gap='large')
    with chart_col1:
        st.subheader("Desempenho Relativo Normalizado")
        if not norm_prices.empty and ordered_cols_for_cards:
            st.line_chart(norm_prices[ordered_cols_for_cards], height=500)
        else: st.info("Dados insuficientes para gráfico de desempenho.")

    with chart_col2:
        st.subheader("Risco vs. Retorno (Anualizado)")
        scatter_data = pd.DataFrame({
            'Volatilidade': vols, 'Retorno': rets_total_period,
            'Sharpe': sharpe_ratios, 'Ativo': vols.index
        }).reindex(ordered_cols_for_cards).dropna()

        if not scatter_data.empty:
            fig = px.scatter(scatter_data, x='Volatilidade', y='Retorno', text='Ativo', color='Sharpe',
                             color_continuous_scale=px.colors.sequential.Bluered_r, hover_name='Ativo', size=np.full(len(scatter_data), 30))
            fig.update_traces(textposition='top center', textfont=dict(color='grey', size=10))
            fig.update_layout(height=500, xaxis_tickformat=".2%", yaxis_tickformat=".2%",
                              xaxis_title="Volatilidade (anualizada)", yaxis_title="Retorno Total (no período)",
                              coloraxis_colorbar_title_text='Sharpe Anual.')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Dados insuficientes para gráfico Risco vs. Retorno.")

# --- Execução Principal do App ---
with st.sidebar:
    st.title("Configurações")
    selected_acoes_no_suffix, selected_acoes_with_suffix, start_date_acoes, end_date_acoes = \
        build_sidebar_generic("Ações", "tickers_ibra.csv", "Selecione as Empresas",
                              "Códigos das Ações", "acoes", LOGO_ACOES_PATH)
    st.divider()
    selected_fiis_no_suffix, selected_fiis_with_suffix, start_date_fiis, end_date_fiis = \
        build_sidebar_generic("Fundos Imobiliários", "tickers_fiis.csv", "Selecione os FIIs",
                              "Códigos dos FIIs", "fiis", LOGO_FIIS_PATH)

prices_acoes_df = None
if selected_acoes_with_suffix:
    prices_acoes_df = load_price_data(selected_acoes_with_suffix, "^BVSP", start_date_acoes, end_date_acoes, "Ações")

prices_fiis_df = None
if selected_fiis_with_suffix:
    prices_fiis_df = load_price_data(selected_fiis_with_suffix, "IFIX.SA", start_date_fiis, end_date_fiis, "FIIs")

st.title('Dashboard de Ações e Fundos Imobiliários')

acoes_data_available = selected_acoes_no_suffix and prices_acoes_df is not None and not prices_acoes_df.empty
fiis_data_available = selected_fiis_no_suffix and prices_fiis_df is not None and not prices_fiis_df.empty

if acoes_data_available:
    build_main_section("Ações", selected_acoes_no_suffix, prices_acoes_df, "IBOV",
                       "Carteira Ações", {'default_ticker_icon_base_url': IMAGES_BASE_URL,
                                         'portfolio': LOGO_ACOES_PATH, 'benchmark': LOGO_ACOES_PATH})
    if fiis_data_available: st.divider()
elif selected_acoes_no_suffix:
    st.warning("Dados de ações não carregados. Verifique período ou tickers.")

if fiis_data_available:
    build_main_section("Fundos Imobiliários", selected_fiis_no_suffix, prices_fiis_df, "IFIX",
                       "Carteira FIIs", {'default_ticker_icon_base_url': IMAGES_BASE_URL,
                                         'portfolio': LOGO_FIIS_PATH, 'benchmark': LOGO_FIIS_PATH})
elif selected_fiis_no_suffix:
     st.warning("Dados de FIIs não carregados. Verifique período ou tickers.")

if not (selected_acoes_no_suffix or selected_fiis_no_suffix):
    st.info("👈 Selecione ativos na barra lateral para visualizar o dashboard.")