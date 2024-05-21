import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime

# Function to fetch stock data
def get_data(tickers, start_date):
    data = yf.download(tickers, start=start_date)['Adj Close']
    return data

# Function to calculate portfolio statistics
def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std
    return returns, std, sharpe_ratio

# Function to minimize negative Sharpe ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    returns, std, sharpe_ratio = portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe_ratio

# Function to get optimal weights
def get_optimal_weights(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Function to calculate efficient frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        returns, std, sharpe_ratio = portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0,i] = returns
        results[1,i] = std
        results[2,i] = sharpe_ratio
        weights_record.append(weights)
    return results, weights_record

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(weights, mean_returns, cov_matrix, num_simulations, num_days):
    portfolio_simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        portfolio_return = np.cumprod(np.dot(daily_returns, weights) + 1)
        portfolio_simulations[i,:] = portfolio_return
    return portfolio_simulations

# Function to plot efficient frontier and the user's portfolio
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios, risk_free_rate, portfolio_return, portfolio_std, optimal_weights):
    results, _ = efficient_frontier(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_return = results[0, max_sharpe_idx]
    max_sharpe_std = results[1, max_sharpe_idx]

    plt.figure(figsize=(10, 7))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
    plt.colorbar(label='Sharpe ratio')
    plt.scatter(max_sharpe_std, max_sharpe_return, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(portfolio_std, portfolio_return, marker='o', color='g', s=200, label='Your Portfolio')

    # Plotting the Capital Market Line (CML)
    cml_x = np.linspace(0, max(results[1]) * 1.1, 100)
    cml_y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_std * cml_x
    plt.plot(cml_x, cml_y, 'r--', label='Capital Market Line')
    
    plt.title('Efficient Frontier')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Return')
    plt.legend(labelspacing=0.8)
    st.pyplot(plt)

    # Display optimal weights
    st.subheader('Optimal Weight Distribution for Maximum Sharpe Ratio')
    optimal_weights_df = pd.DataFrame(optimal_weights, index=tickers, columns=['Weight'])
    st.write(optimal_weights_df)

# Streamlit app interface
st.title('Portfolio Analyzer')

# User input for stocks and weights
st.header('Select up to 15 stocks and their portfolio percentages:')
num_stocks = st.number_input('Number of stocks in portfolio:', min_value=1, max_value=15, value=5)

tickers = []
weights = []
for i in range(num_stocks):
    ticker = st.text_input(f'Stock {i+1} ticker:')
    weight = st.number_input(f'Stock {i+1} portfolio percentage:', min_value=0.0, max_value=100.0, value=10.0)
    tickers.append(ticker)
    weights.append(weight)

start_date = st.date_input('Select start date for historical data (from 2007 onwards):', min_value=datetime(2007, 1, 1), value=datetime(2010, 1, 1))
risk_free_rate = st.number_input('Risk-free rate (as a decimal):', min_value=0.0, max_value=0.1, value=0.01)

if sum(weights) != 100:
    st.warning('Portfolio percentages must sum to 100.')
else:
    data = get_data(tickers, start_date)
    log_returns = np.log(data / data.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()

    # Calculate portfolio statistics
    weights = np.array(weights) / 100
    portfolio_return, portfolio_std, portfolio_sharpe_ratio = portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)

    # Find optimal weights
    optimal_weights = get_optimal_weights(mean_returns, cov_matrix, risk_free_rate)
    optimal_return, optimal_std, optimal_sharpe_ratio = portfolio_statistics(optimal_weights, mean_returns, cov_matrix, risk_free_rate)

    st.write(f'Expected Portfolio Return: {portfolio_return:.2f}')
    st.write(f'Portfolio Standard Deviation: {portfolio_std:.2f}')
    st.write(f'Portfolio Sharpe Ratio: {portfolio_sharpe_ratio:.2f}')
    st.write(f'Optimal Portfolio Return: {optimal_return:.2f}')
    st.write(f'Optimal Portfolio Standard Deviation: {optimal_std:.2f}')
    st.write(f'Optimal Portfolio Sharpe Ratio: {optimal_sharpe_ratio:.2f}')

    st.markdown("""
        **Sharpe Ratio:** The Sharpe ratio is a measure of risk-adjusted return. It is calculated as the difference between the portfolio return and the risk-free rate, divided by the portfolio's standard deviation. A higher Sharpe ratio indicates a better risk-adjusted return.
    """)

    # Correlation Matrix
    st.header('Correlation Matrix')
    st.markdown("""
        The correlation matrix shows the correlation coefficients between the returns of the selected stocks. 
        A correlation coefficient close to 1 indicates that the stocks move in the same direction, while a coefficient close to -1 indicates they move in opposite directions.
    """)

    corr_matrix = log_returns.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

    # Portfolio Holdings Pie Chart
    st.header('Portfolio Holdings')
    st.markdown("""
        This pie chart represents the allocation of your portfolio based on the weights you have provided. 
        It helps visualize the distribution of investments across the selected stocks.
    """)

    fig, ax = plt.subplots()
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Monte Carlo Simulation
    st.header('Monte Carlo Simulation')
    st.markdown("""
        The Monte Carlo simulation uses historical data to model the future performance of your portfolio under different scenarios. 
        It helps assess the potential impact of market volatility and crises on your portfolio's value over time.
    """)

    num_simulations = 1000
    num_days = 252
    portfolio_simulations = monte_carlo_simulation(weights, mean_returns, cov_matrix, num_simulations, num_days)
    
    fig, ax = plt.subplots()
    for i in range(num_simulations):
        ax.plot(portfolio_simulations[i, :], color='lightblue', alpha=0.05)
    ax.plot(portfolio_simulations.mean(axis=0), color='blue', label='Mean Portfolio Value')
    ax.set_title('Monte Carlo Simulation of Portfolio Performance')
    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    st.pyplot(fig)


    sp500 = get_data(['^GSPC'], start_date)
    portfolio_values = (data / data.iloc[0]) @ weights
    sp500_values = sp500 / sp500.iloc[0]

    fig, ax = plt.subplots()
    portfolio_values.plot(ax=ax, label='Portfolio')
    sp500_values.plot(ax=ax, label='S&P 500')



    # Plot the efficient frontier and the user's portfolio
    def plot_efficient_frontier_with_optimal(mean_returns, cov_matrix, num_portfolios, risk_free_rate, portfolio_return, portfolio_std, optimal_weights):
        results, weights_record = efficient_frontier(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
        st.subheader('Efficient Frontier & Capital Market Line')
        st.markdown("""
        **Efficient Frontier:** The efficient frontier represents the set of optimal portfolios that offer the highest expected return for a defined level of risk. 
        Portfolios that lie below the efficient frontier are sub-optimal because they do not provide enough return for the level of risk.
    """)
        st.markdown("""
        **Capital Market Line (CML):** The CML represents portfolios that optimally combine risk and return. 
        It is a straight line that starts from the risk-free rate and extends tangentially to the efficient frontier.
    """)
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_std = results[1, max_sharpe_idx]

        plt.figure(figsize=(10, 7))
        plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
        plt.colorbar(label='Sharpe ratio')
        plt.scatter(max_sharpe_std, max_sharpe_return, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
        plt.scatter(portfolio_std, portfolio_return, marker='o', color='g', s=200, label='Your Portfolio')

        # Plotting the Capital Market Line (CML)
        cml_x = np.linspace(0, max(results[1]) * 1.1, 100)
        cml_y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_std * cml_x
        plt.plot(cml_x, cml_y, 'r--', label='Capital Market Line')

        plt.title('Efficient Frontier')
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return')
        plt.legend(labelspacing=0.8)
        st.pyplot(plt)


        # Display optimal weights
        st.subheader('Optimal Weight Distribution for Maximum Sharpe Ratio')
        optimal_weights_df = pd.DataFrame(optimal_weights, index=tickers, columns=['Weight'])
        st.write(optimal_weights_df)
        st.markdown("""
        **Optimal Portfolio:** The optimal portfolio on the efficient frontier is the one that provides the highest Sharpe ratio, representing the best trade-off between risk and return.
    """)

        st.markdown("""
        **Sharpe Ratio:** The Sharpe ratio is a measure of risk-adjusted return. It is calculated as the difference between the portfolio return and the risk-free rate, divided by the portfolio's standard deviation. A higher Sharpe ratio indicates a better risk-adjusted return.
    """)


    plot_efficient_frontier_with_optimal(mean_returns, cov_matrix, 10000, risk_free_rate, portfolio_return, portfolio_std, optimal_weights)
    
    # Performance comparison between the user's portfolio and the S&P 500
    portfolio_cumulative_returns = (1 + log_returns @ weights).cumprod()
    sp500_cumulative_returns = (1 + np.log(sp500 / sp500.shift(1)).dropna()).cumprod()

    fig, ax = plt.subplots()
    portfolio_cumulative_returns.plot(ax=ax, label='Portfolio')
    sp500_cumulative_returns.plot(ax=ax, label='S&P 500')
    ax.legend()
    ax.set_title('Portfolio Performance vs S&P 500')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    st.pyplot(fig)


    st.markdown("""
        **Portfolio Performance vs S&P 500:** This graph shows the performance of your portfolio compared to the S&P 500 index since the selected start date. 
        It helps visualize how your portfolio has performed relative to the overall market.
    """)
