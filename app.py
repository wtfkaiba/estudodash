import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

st.set_page_config(layout="wide")


def build_sidebar():
    st.image("images/logo.png", width=60)
    ticker_list = pd.read_csv("tickers_ibra.csv", index_col=0)
    tickers = st.multiselect(label="Selecione as Empresas", options=ticker_list, placeholder='Códigos')
    tickers = [t+".SA" for t in tickers]
    start_date = st.date_input("De", format="DD/MM/YYYY", value=datetime(2024,1,2), key="acoes")
    end_date = st.date_input("Até", format="DD/MM/YYYY", value="today", key="acoesend")

    if tickers:
        prices = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
        if len(tickers) ==1:
            prices = prices.to_frame()
            prices.columns = [tickers[0].rstrip(".SA")]

        prices.columns = prices.columns.str.rstrip(".SA")
        prices['IBOV'] = yf.download("^BVSP", start=start_date, end=end_date)['Adj Close']
        return tickers, prices
    return None, None

def build_sidebar_fiis():
    st.image("images/fiilogo.png", width=60)
    ticker_list_fii = pd.read_csv("tickers_fiis.csv", index_col=0)
    tickers_fiis = st.multiselect(label="Selecione os Fundos Imobiliários", options=ticker_list_fii, placeholder='Códigos')
    tickers_fiis = [t+".SA" for t in tickers_fiis]
    start_date_fii = st.date_input("De", format="DD/MM/YYYY", value=datetime(2024,1,2), key="fiis")
    end_date_fii = st.date_input("Até", format="DD/MM/YYYY", value="today", key="fiisend")

    if tickers_fiis:
        prices_fiis = yf.download(tickers_fiis, start=start_date_fii, end=end_date_fii)["Adj Close"]
        if len(tickers_fiis) ==1:
            prices_fiis = prices_fiis.to_frame()
            prices_fiis.columns = [tickers_fiis[0].rstrip(".SA")]

        prices_fiis.columns = prices_fiis.columns.str.rstrip(".SA")
        prices_fiis['IFIX'] = yf.download("IFIX.SA", period="1d")['Adj Close']
        return tickers_fiis, prices_fiis
    return None, None

def build_main(tickers, prices):
    weights = np.ones(len(tickers))/len(tickers)
    prices['portfolio'] = prices.drop("IBOV", axis=1) @ weights
    norm_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change()[1:]
    vols = returns.std()*np.sqrt(252)
    rets = (norm_prices.iloc[-1] - 100) / 100

    mygrid = grid(5 ,5 ,5 ,5 ,5 ,5, vertical_align="top")
    for t in prices.columns:
        c = mygrid.container(border=True)
        c.subheader(t, divider="red")
        colA, colB, colC = c.columns(3)
        if t == "portfolio":
            colA.image("images/logo.png", width=85)
        elif t == "IBOV":
            colA.image("images/logo.png", width=85)
            
        else:
            colA.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=85)
        colB.metric(label="retorno", value=f"{rets[t]:.0%}")
        colC.metric(label="volatilidade", value=f"{vols[t]:.0%}")
        style_metric_cards(background_color='rgba(255,255,255,0)')

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader("Desempenho Relativo")
        st.line_chart(norm_prices, height=600)

    with col2:
        st.subheader("Risco-Retorno")
        fig = px.scatter(
            x=vols,
            y=rets,
            text=vols.index,
            color=rets/vols,
            color_continuous_scale=px.colors.sequential.Bluered_r
        )
        fig.update_traces(
            textfont_color='white', 
            marker=dict(size=45),
            textfont_size=10,                  
        )
        fig.layout.yaxis.title = 'Retorno Total'
        fig.layout.xaxis.title = 'Volatilidade (anualizada)'
        fig.layout.height = 600
        fig.layout.xaxis.tickformat = ".0%"
        fig.layout.yaxis.tickformat = ".0%"        
        fig.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(fig, use_container_width=True)
    # st.dataframe(prices)


def build_main_fii(tickers_fiis, prices_fiis):
    weights_fii = np.ones(len(tickers_fiis))/len(tickers_fiis)
    prices_fiis['portfolio FIIs'] = prices_fiis.drop("IFIX", axis=1) @ weights_fii
    norm_prices_fii = 100 * prices_fiis / prices_fiis.iloc[0]
    returns_fii = prices_fiis.pct_change()[1:]
    vols_fii = returns_fii.std()*np.sqrt(252)
    rets_fii = (norm_prices_fii.iloc[-1] - 100) / 100

    mygrid_fii = grid(5 ,5 ,5 ,5 ,5 ,5, vertical_align="top")
    for t in prices_fiis.columns:
        c = mygrid_fii.container(border=True)
        c.subheader(t, divider="red")
        colA, colB, colC = c.columns(3)
        if t == "portfolio FIIs":
            colA.image("images/fiilogo.png", width=85)
        elif t == "IFIX":
            colA.image("images/fiilogo.png", width=85)
            
        else:
            colA.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=85)
        colB.metric(label="retorno", value=f"{rets_fii[t]:.0%}")
        colC.metric(label="volatilidade", value=f"{vols_fii[t]:.0%}")
        style_metric_cards(background_color='rgba(255,255,255,0)')

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader("Desempenho Relativo")
        st.line_chart(norm_prices_fii, height=600)

    with col2:
        st.subheader("Risco-Retorno")
        fig_fii = px.scatter(
            x=vols_fii,
            y=rets_fii,
            text=vols_fii.index,
            color=rets_fii/vols_fii,
            color_continuous_scale=px.colors.sequential.Bluered_r
        )
        fig_fii.update_traces(
            textfont_color='white', 
            marker=dict(size=45),
            textfont_size=10,                  
        )
        fig_fii.layout.yaxis.title = 'Retorno Total'
        fig_fii.layout.xaxis.title = 'Volatilidade (anualizada)'
        fig_fii.layout.height = 600
        fig_fii.layout.xaxis.tickformat = ".0%"
        fig_fii.layout.yaxis.tickformat = ".0%"        
        fig_fii.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(fig_fii, use_container_width=True)
    # st.dataframe(prices)

    
with st.sidebar:
    tickers, prices = build_sidebar()
    tickers_fiis, prices_fiis = build_sidebar_fiis()

st.title('Ações e FIIs Dashboard')
if tickers:
    build_main(tickers, prices)

if tickers_fiis:
    build_main_fii(tickers_fiis, prices_fiis)