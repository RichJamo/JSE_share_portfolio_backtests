# JSE_share_portfolio_backtests
This is the code for a backtesting engine that I built in 2016/17 for Dalebrook Capital.

Run **backtest_main_program.py** in the /backtest folder, to see it in action.

It references multiple csv datafiles, which are included in the repository in backtest/csv_dataset3.

The methodology here is to start by importing data for the whole universe of shares being traded on the JSE over the last 20 years (approximately).
Multiple datasets are imported for each share - both technical market data (price, volume, market cap), as well as fundamental financial data (drawn from financial statements).
These are then combined into ND data sets, with one for example being monthly price over a number of years for every ticker considered.

Portfolio construction is done by a process of elimination, through a number of screens.
The first is a forensic screen - looking at financial data to screen out the 10-15% of companies who's financials suggest that we should have a low level of confidence in their honesty/credibility.
The second is a value screen - screening to select for those companies who's price to EBITDA (or similar metric) indicates that they represent good value at current price.
The third is a quality screen - looking at financial data to select those companies who might have better 'quality' earnings - that will sustain over time.
(This approach was based on the book, Quantitative Value)

The portfolio is then weighted, with a choice of several weighting options:
- equal weighting
- market cap weighting
- inverse volatility weighting

This weighting is then combined with a limit on how big or small any one position can get - usually 2% and 10%.

One then ends up with a portfolio of around 20 stocks, depending on the parameters chosen along the way.

It is fairly easy to backtest and plot visually how these different portfolios would have performed, given the data structure used. One can mix and match the different parameters and strategies to see effects.
The plots include a number of different metrics, including: CAGR, Sharpe ratio, Sortino ratio, worst month, max drawdown, etc.

A second strategy is also implemented here, called Quantititave Momentum, also based on a book of the same name, by the same authors. A detailed article on this strategy can be found here: https://alphaarchitect.com/2015/12/01/quantitative-momentum-investing-philosophy/

The key steps are the following: 
1 - Identify Universe: Our universe in this case were all JSE (Johannesburg stock exhange) exchange-traded stocks - about 400 in total.
2 - Remove Outliers: We leverage algos derived from academic research to eliminate stocks with red flags.
3 - Screen for Momentum: We screen for stocks with the strongest momentum. (Top 50 stocks)
4 - Screen for Momentum Quality: We seek high-quality momentum action that is less dependent on large return gaps. (Top 25 stocks)

Still to do:
- a front end that makes it easier to change parameters and strategy combinations
- iron out some compatibility bugs that have cropped up since I first created this
