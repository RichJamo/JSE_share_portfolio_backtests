# JSE_share_portfolio_backtests
This is the code for our current backtesting engine on a JSE share portfolio.
Run backtest_main_program.py in the /backtest folder, to see it in action.
It references multiple csv datafiles, which are included in the repository.

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

A second strategy is also implemented here, called Quantititave Momentum, also based on a book of the same name, by the same authors.

Still to do:
- a front end that makes it easier to change parameters and strategy combinations
