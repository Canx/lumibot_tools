from finvizfinance.screener.overview import Overview

foverview = Overview()
filters_dict = {'Index':'S&P 500','Sector':'Basic Materials'}
foverview.set_filter(filters_dict=filters_dict)
df = foverview.screener_view()
df.head()
